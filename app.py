
import numpy as np
import torch
from torch.utils.data import DataLoader
from open3d_wrapper import Open3DWrapper
from omkar.pose_prediction.pose_predictor import SeSGCNPosePredictor
from omkar.pose_prediction.utils.datasets.CHICO import PoseDataset, normal_actions
'''
Shows animation for of every subject doing every action
'''


def predict(person_kpts):
    return person_kpts

def get_sequence(chico_poses_path):
    chico_dataset = PoseDataset(
        chico_poses_path,
        'test',
        10,
        25,
        actions = normal_actions,
        win_stride = 1
       )

    chico_dataset = DataLoader(
        chico_dataset, 
        batch_size = 1, 
        shuffle=True,
        num_workers=0,
        pin_memory=True)

    for i, batch in enumerate(chico_dataset):
        if i == 100:
            return batch
   

def main():
    FPS = 2

    wrapper = Open3DWrapper()
    wrapper.initialize_visualizer()

    # all_actions = CHICODataset.actions
    all_actions = [
        "span_light",
        "span_light_CRASH",
        "hammer",
        "lift",
        "place-hp",
        "place-hp_CRASH",
        "place-lp",
        "place-lp_CRASH",
        "polish",
        "polish_CRASH",
        "span_heavy",
        "span_heavy_CRASH",
    ]

    keypoints_links = [
        [0,1],
        [1,2],
        [2,3],
        [0,4],
        [4,5],
        [5,6],
        [1,9],
        [4,12],
        [8,7],
        [8,9],
        [8,12],
        [9,10],
        [10,11],
        [12,13],
        [13,14]
    ]
    
    sequence =  get_sequence("/home/prakhar/ws/cobot/human-pose-prediction/data/chico/poses")

    input_sequence = sequence[:, 0:10, :, :].permute(0,3,1,2).float()
    input_sequence_copy = sequence[:, 0:10, :, :].detach().clone()

    predictor = SeSGCNPosePredictor()
    predictor.load_config("/home/prakhar/ws/cobot/human-pose-prediction/omkar/config/model_chico_3d_25frames_student_config.ini")
    predictor.create_model("/home/prakhar/ws/cobot/human-pose-prediction/omkar/data/checkpoints/chico/chico_3d_25frames_Student")
    predictor.load_masks(
        maskA_path= "/home/prakhar/ws/cobot/human-pose-prediction/omkar/data/checkpoints/chico/masks/maskA_25fps.npy",
        maskT_path= "/home/prakhar/ws/cobot/human-pose-prediction/omkar/data/checkpoints/chico/masks/maskT_25fps.npy"
    )

    predicted_sequence = predictor.predict(input_sequence)
    print(predicted_sequence.shape)

    final_predicted_sequence = torch.cat((input_sequence_copy, predicted_sequence.to('cpu')),1)
    print(final_predicted_sequence.shape)

    
    final_predicted_sequence = final_predicted_sequence.cpu().detach().numpy()[0]
    ground_truth_sequence = sequence.cpu().detach().numpy()[0]

    print(final_predicted_sequence.shape)
    print(ground_truth_sequence.shape)

    skeleton, predicted_skeleton = None, None
    coordinate_system = None

    for i, (ground_truth_frame, predicted_frame) in enumerate(zip(ground_truth_sequence, final_predicted_sequence)):
        
        # ground truth skeleton
        person_kpts = ground_truth_frame
        if skeleton is None:
            skeleton = wrapper.create_skeleton(person_kpts, keypoints_links, radius=10)
        else:
            skeleton.update(person_kpts)

        # testing prediction skeleton
        predicted_kpts = predicted_frame

        predicted_line_colors = [[ 125/ 255, 0 / 255, 0 / 255]] * len(predicted_kpts)

        predicted_point_colors = [[ 255/ 255, 0 / 255, 0 / 255]] * len(predicted_kpts)



        # if rendering in same scene, then translating predicted points
        location = [0,0,0]
        translated_predicted_kpts = [ np.array(point) for point in predicted_kpts ]
        #translated_predicted_kpts = [ point + np.random.rand(1,3)[0] for point in translated_predicted_kpts ]

        translated_predicted_kpts = [ point + np.array(location) for point in translated_predicted_kpts ]

        if predicted_skeleton is None:
            predicted_skeleton = wrapper.create_skeleton( 
                translated_predicted_kpts, 
                keypoints_links, 
                radius=10, 
                line_colors = predicted_line_colors,
                point_colors = predicted_point_colors )
        # elif i == 10:
        #     predicted_skeleton = wrapper.create_skeleton( 
        #         translated_predicted_kpts, 
        #         keypoints_links, 
        #         radius=10, 
        #         line_colors = [[ 125/ 255, 0 / 255, 0 / 255]] * len(predicted_kpts),
        #         point_colors = [[ 255/ 255, 125 / 255, 125 / 255]] * len(predicted_kpts) )
        else:
                        
            predicted_skeleton.update(translated_predicted_kpts)

                    # # Create / Update robot skeleton
                    # if robot is None:
                    #     robot = wrapper.create_skeleton(
                    #         robot_kpts,
                    #         kuka_links,
                    #         radius=20,
                    #         line_colors=[[0, 0, 1]] * 9,
                    #         point_colors=[[0, 0, 1]] * 9,
                    #     )
                    # else:
                    #     robot.update(robot_kpts)

        # Add coordinate system
        if coordinate_system is None:
            pts = np.asarray(person_kpts)
            loc = [
                    np.min(pts[:, 0]).item(),
                    np.min(pts[:, 1]).item(),
                    np.min(pts[:, 2]).item(),
                    ]
            coordinate_system = wrapper.create_coordinate_system(loc, 150)

        wrapper.update()
        wrapper.wait(1 / FPS)

    wrapper.clear()

if __name__ == "__main__":
    main()
