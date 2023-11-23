import myosuite
import mujoco
from mujoco import viewer

import gym
import mediapy as media
import matplotlib.pyplot as plt
import cv2


from myosuite.physics.sim_scene import SimScene
import myosuite.utils.import_utils as import_utils


path_created = "C:/Users/zeive/anaconda3/Lib/site-packages/myosuite/simhive/myo_sim/ampModel/OSL_TFA_L_Ankle_Modified_cvt3.xml"
path_original = "C:/Users/zeive/anaconda3/Lib/site-packages/myosuite/simhive/myo_sim/leg/myolegs_v0.54(mj236).mjb"
path_hand = "C:/Users/zeive/anaconda3/Lib/site-packages/myosuite/simhive/myo_sim/hand/myohand.xml"
model = mujoco.MjModel.from_xml_path(path_created)
model1 = mujoco.MjModel.from_binary_path(path_original)
data1 = mujoco.MjData(model1)
hand_model = mujoco.MjModel.from_xml_path(path_hand)
hand_data = mujoco.MjData(hand_model)


simEnv = SimScene.get_sim(path_original)
simEnv.forward()
try:
    simEnv.model.geom()
except KeyError as e:
    print(e)

print('raw access:\n', simEnv.data.geom_xpos)

options = mujoco.MjvOption()


#window = viewer.launch_passive(simEnv.model.ptr, simEnv.data.ptr)
#Access to options
#window.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
#window.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True


frame = simEnv.renderer.render_offscreen(width=640,
                        height=480)


cv2.imshow("img", frame)


cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image

# try:
#    model.geom()
# except KeyError as e:
#    print(e)

# try:
#    model1.geom()
# except KeyError as e:
#    print(e)

# for i in range (model.ngeom):
#    print('name of geom ', i, ' : ', model.geom(i).name)

# data = mujoco.MjData(model)
# print(data.geom_xpos)

# mujoco.mj_kinematics(model, data)
# print('raw access:\n', data.geom_xpos)

# MjData also supports named access:
# print('\nnamed access:\n', data.geom('AnkleAssembly_geom_1').xpos)


# for i in range (model1.ngeom):
#    print('name of geom ', i, ' : ', model1.geom(i).name)
