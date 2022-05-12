import os
from pathlib import Path

def meshlab_resave(modelname, outputname):
    os.chdir("C:/Program Files/VCG/MeshLab/")
    meshlabserver = r"meshlabserver.exe"
    cmd = meshlabserver + " -i " + str(modelname) + " -o " + str(outputname) +" -m vc vn"
    os.system(cmd)

def meshlab_wscript(modelname, outputname):
    os.chdir("C:/Program Files/VCG/MeshLab/")
    meshlabserver = r"meshlabserver.exe"
    scriptname = " -s D:/code/demo/stl/PLT_TO_STL.mlx"
    cmd = meshlabserver + " -i " + str(modelname) + " -o " + str(outputname) +" -m vc vn"  + str(scriptname)
    os.system(cmd)

meshlab_wscript(r"D:\code\demo\stl\liver_cut1.ply",r"D:\code\demo\stl\liver_cut1.stl")
meshlab_wscript(r"D:\code\demo\stl\liver_remain1.ply",r"D:\code\demo\stl\liver_remain1.stl")
meshlab_wscript(r"D:\code\demo\stl\g_vessel1.ply",r"D:\code\demo\stl\g_vessel1.stl")
meshlab_wscript(r"D:\code\demo\stl\m_vessel1.ply",r"D:\code\demo\stl\m_vessel1.stl")
meshlab_wscript(r"D:\code\demo\stl\tumor1.ply",r"D:\code\demo\stl\tumor1.stl")