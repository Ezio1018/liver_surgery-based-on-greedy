import pyvista as pv
# i = 5
# surface1 = pv.read('finaldata/肝实质.stl')
# surface2 = pv.read('finaldata/门静脉.stl')
# surface3 = pv.read('finaldata/肝静脉.stl')

# p = pv.Plotter()
# p.set_background('white')
# p.add_mesh(surface1,color='blue',opacity=0.5)
# p.add_mesh(surface2,color='purple',opacity=0.8)
# p.add_mesh(surface3,color='green',opacity=0.8)

# p.show()
def visualize(i):
    surface1 = pv.read('finaldata/liver_remain'+str(i)+'.stl')
    surface2 = pv.read('finaldata/m_vessel'+str(i)+'.stl')
    surface3 = pv.read('finaldata/g_vessel'+str(i)+'.stl')
    surface4 = pv.read('finaldata/liver_cut'+str(i)+'.stl')
    surface5 = pv.read('finaldata/tumor1.stl')

    p = pv.Plotter()
    p.set_background('white')
    p.add_mesh(surface1,color='#FFD2E9',opacity=0.6)
    p.add_mesh(surface2,color='#55AAFF',opacity=0.5)
    p.add_mesh(surface3,color='0000FF',opacity=0.5)
    p.add_mesh(surface4,color='#FFFF80',opacity=0.5)
    p.add_mesh(surface5,color='red',opacity=1)
    p.show()

visualize(1)