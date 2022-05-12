import trimesh

def smooth(i):
    i=str(i)
    a = trimesh.load_mesh("stl/g_vessel"+i+".stl")
    trimesh.smoothing.filter_humphrey(a)
    a.export("finaldata/g_vessel"+i+".stl")
    a = trimesh.load_mesh("stl/m_vessel"+i+".stl")
    trimesh.smoothing.filter_humphrey(a)
    a.export("finaldata/m_vessel"+i+".stl")
    a = trimesh.load_mesh("stl/liver_remain"+i+".stl")
    trimesh.smoothing.filter_humphrey(a)
    a.export("finaldata/liver_remain"+i+".stl")
    a = trimesh.load_mesh("stl/liver_cut"+i+".stl")
    trimesh.smoothing.filter_humphrey(a)
    a.export("finaldata/liver_cut"+i+".stl")

    a = trimesh.load_mesh("stl/tumor"+i+".stl")
    trimesh.smoothing.filter_humphrey(a)
    a.export("finaldata/tumor"+i+".stl")
smooth(1)