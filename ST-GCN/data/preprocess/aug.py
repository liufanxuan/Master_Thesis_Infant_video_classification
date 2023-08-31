from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0,"/home/fanxuan/stam-main/src/")
from data.preprocess.data_utils import get_video_information, load_raw_pkl_files, edit_df, get_skel, \
    interpolate_df, smooth, normalise_skeletons, get_joint_angles, get_dynamics_xy, get_dynamics_angle
from utils.utils import ConfigLoader
from timeit import default_timer as timer



def flip_pose(src_infant_dir, dst_infant_dir, infant_id):
    df = pd.read_pickle(src_infant_dir)
    if df['pixel_x'][1] == 1280:
        df['x'] = pd.to_numeric(df['x'])
        df['x'] = pd.to_numeric(df['pixel_x'])-df['x']
        RShoulder = df['x'][df['bp']=="RShoulder"].to_numpy()
        LShoulder = df['x'][df['bp']=="LShoulder"].to_numpy()
        RElbow = df['x'][df['bp']=="RElbow"].to_numpy()
        LElbow = df['x'][df['bp']=="LElbow"].to_numpy()
        RWrist = df['x'][df['bp']=="RWrist"].to_numpy()        
        LWrist = df['x'][df['bp']=="LWrist"].to_numpy() 
        RHip = df['x'][df['bp']=="RHip"].to_numpy() 
        LHip = df['x'][df['bp']=="LHip"].to_numpy()
        RKnee = df['x'][df['bp']=="RKnee"].to_numpy() 
        LKnee = df['x'][df['bp']=="LKnee"].to_numpy()  
        RAnkle = df['x'][df['bp']=="RAnkle"].to_numpy() 
        LAnkle = df['x'][df['bp']=="LAnkle"].to_numpy()
        REye = df['x'][df['bp']=="REye"].to_numpy()
        LEye = df['x'][df['bp']=="LEye"].to_numpy()
        REar = df['x'][df['bp']=="REar"].to_numpy()
        LEar = df['x'][df['bp']=="LEar"].to_numpy()
        df['x'][df['bp']=="RShoulder"] = LShoulder
        df['x'][df['bp']=="LShoulder"] = RShoulder
        df['x'][df['bp']=="RElbow"] = LElbow
        df['x'][df['bp']=="LElbow"] = RElbow
        df['x'][df['bp']=="RWrist"] = LWrist
        df['x'][df['bp']=="LWrist"] = RWrist
        df['x'][df['bp']=="RHip"] = LHip
        df['x'][df['bp']=="LHip"] = RHip
        df['x'][df['bp']=="RKnee"] = LKnee
        df['x'][df['bp']=="LKnee"] = RKnee
        df['x'][df['bp']=="RAnkle"] = LAnkle
        df['x'][df['bp']=="LAnkle"] = RAnkle
        df['x'][df['bp']=="REye"] = LEye
        df['x'][df['bp']=="LEye"] = REye
        df['x'][df['bp']=="REar"] = LEar
        df['x'][df['bp']=="LEar"] = REar
        RShoulder = df['y'][df['bp']=="RShoulder"].to_numpy()
        LShoulder = df['y'][df['bp']=="LShoulder"].to_numpy()
        RElbow = df['y'][df['bp']=="RElbow"].to_numpy()
        LElbow = df['y'][df['bp']=="LElbow"].to_numpy()
        RWrist = df['y'][df['bp']=="RWrist"].to_numpy()        
        LWrist = df['y'][df['bp']=="LWrist"].to_numpy() 
        RHip = df['y'][df['bp']=="RHip"].to_numpy() 
        LHip = df['y'][df['bp']=="LHip"].to_numpy()
        RKnee = df['y'][df['bp']=="RKnee"].to_numpy() 
        LKnee = df['y'][df['bp']=="LKnee"].to_numpy()  
        RAnkle = df['y'][df['bp']=="RAnkle"].to_numpy() 
        LAnkle = df['y'][df['bp']=="LAnkle"].to_numpy()
        REye = df['y'][df['bp']=="REye"].to_numpy()
        LEye = df['y'][df['bp']=="LEye"].to_numpy()
        REar = df['y'][df['bp']=="REar"].to_numpy()
        LEar = df['y'][df['bp']=="LEar"].to_numpy()
        df['y'][df['bp']=="RShoulder"] = LShoulder
        df['y'][df['bp']=="LShoulder"] = RShoulder
        df['y'][df['bp']=="RElbow"] = LElbow
        df['y'][df['bp']=="LElbow"] = RElbow
        df['y'][df['bp']=="RWrist"] = LWrist
        df['y'][df['bp']=="LWrist"] = RWrist
        df['y'][df['bp']=="RHip"] = LHip
        df['y'][df['bp']=="LHip"] = RHip
        df['y'][df['bp']=="RKnee"] = LKnee
        df['y'][df['bp']=="LKnee"] = RKnee
        df['y'][df['bp']=="RAnkle"] = LAnkle
        df['y'][df['bp']=="LAnkle"] = RAnkle
        df['y'][df['bp']=="REye"] = LEye
        df['y'][df['bp']=="LEye"] = REye
        df['y'][df['bp']=="REar"] = LEar
        df['y'][df['bp']=="LEar"] = REar


    df.to_pickle(Path(dst_infant_dir, infant_id[:-8]+'.pkl'))

def main():
    conf = ConfigLoader().config
    data_root = '/home/fanxuan/Open-Pose-Keras/dataset'
    print(f"Start preprocessing poses. Data root: {data_root}.")
    src_data_path = Path('/home/fanxuan/Open-Pose-Keras/dataset/flipped/')
    print(src_data_path)
    dst_data_path = Path(data_root, 'flipped')
    src_infant_dirs = [e for e in Path(src_data_path).rglob('*.pkl')]
    src_infant_dirs = sorted(src_infant_dirs)
    print(src_infant_dirs)

    for i, src_infant_dir in enumerate(src_infant_dirs):
        infant_id = str(src_infant_dir.stem)
        if infant_id == "video_info":
            continue
        print(infant_id)
        dst_infant_dir = dst_data_path
        start_time = timer()
        flip_pose(src_infant_dir, dst_infant_dir, infant_id)
        end_time = timer()
        elapsed_time = end_time - start_time
        print(f"({i + 1:03}/{len(src_infant_dirs):03}): {infant_id} done! Elapsed time: {elapsed_time:.1f} sec.")

if __name__ == '__main__':
    main()