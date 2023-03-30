# Mars rover path solver  - Miguel Perez
import numpy as np
from simpleai.search import astar, SearchProblem
import time
import plotly.graph_objects as px

#------------------------------------------------------------------------------
#   Class definition
#------------------------------------------------------------------------------

class Map(SearchProblem):
    """ The states are represented by a tuple (a,b) where a and b are the yx coordinates. """
    def __init__(self, start, end, mars_map):        
        SearchProblem.__init__(self, start)
        self.goal = end 
        self.mars_map = mars_map

    def actions(self, state):
        act = []
        # Solo podemos movernos a coordenadas adyacentes que tienen una diferencia de altura <0.25
        # Sabemos que la altura mínima es 0 entonces cualquier diferencia con un -1 es >0.25 m
        if abs(mars_map[state[0],state[1]] - mars_map[state[0],state[1]+1]) < 0.25:
            act.append((0,1))
        if abs(mars_map[state[0],state[1]] - mars_map[state[0]+1,state[1]+1]) < 0.25:
            act.append((1,1))
        if abs(mars_map[state[0],state[1]] - mars_map[state[0]+1,state[1]]) < 0.25:
            act.append((1,0))
        if abs(mars_map[state[0],state[1]] - mars_map[state[0]+1,state[1]-1]) < 0.25:
            act.append((1,-1))
        if abs(mars_map[state[0],state[1]] - mars_map[state[0],state[1]-1]) < 0.25:
            act.append((0,-1))
        if abs(mars_map[state[0],state[1]] - mars_map[state[0]-1,state[1]-1]) < 0.25:
            act.append((-1,-1))
        if abs(mars_map[state[0],state[1]] - mars_map[state[0]-1,state[1]]) < 0.25:
            act.append((-1,0))
        if abs(mars_map[state[0],state[1]] - mars_map[state[0]-1,state[1]+1]) < 0.25:
            act.append((-1,1))

        return act
        
    def result(self, state, action):
        new_state = (state[0]+action[0], state[1]+action[1])
        return new_state	
        
    def is_goal(self, state):
        return state == self.goal

    def cost(self, state, action, state2):
        
        """ This method receives two states and an action, and returns the cost of applying the 
            action from the first state to the second state."""

        if action == (0,1) or action == (1,0) or action == (0,-1) or action == (-1,0):
            return 1
        else:
            return 2**(1/2)

    def heuristic(self, state):
        distance = ((self.goal[0]-state[0])**2 + (self.goal[1]-state[1])**2)**(1/2)
        return distance
    
    def show(self, path):
        scale = 10.0174
        x = scale*np.arange(mars_map.shape[1])
        y = scale*np.arange(mars_map.shape[0])
        X, Y = np.meshgrid(x, y)
        path_x, path_y, path_z = np.transpose(path)

        fig = px.Figure(data=[
            px.Surface(
                x=X, y=Y, z=np.flipud(mars_map), colorscale='hot', cmin=0, 
                lighting=dict(ambient=0.0, diffuse=0.8, fresnel=0.02, roughness=0.4, specular=0.2),
                lightposition=dict(x=0, y=nr/2, z=2*mars_map.max())
            ),
            px.Scatter3d(
                x=path_x, y=path_y, z=path_z, name='path', mode='markers', 
                marker=dict(color=np.linspace(0, 1, len(path_x)), colorscale="Bluered", size=4)
            )
        ], layout=px.Layout(
            scene_aspectmode='manual', 
            scene_aspectratio=dict(x=1, y=nr/nc, z=max(mars_map.max()/x.max(), 0.2)), 
            scene_zaxis_range=[0, mars_map.max()]
        ))

        fig.show()

#------------------------------------------------------------------------------
#   Program
#------------------------------------------------------------------------------

mars_map = np.load('mars_map.npy')
nr, nc = mars_map.shape

# Start coordinates
x, y = 2850, 6400 
row = nr - round(y / 10.0174)
col = round(x / 10.0174)
start = (row, col)

# Goal coordinates
x, y = 3150, 6800 
row = nr - round(y / 10.0174)
col = round(x / 10.0174)
goal = (row, col)

my_map = Map(start,goal, mars_map)

# Solve problem using A*
time_start = time.time()
result = astar(my_map, graph_search=True)
time_end = time.time()

print('Goal achieved! \n')
print("A-star search time: ", time_end-time_start, '\n')

#------------------------------------------------------------------------------
#   Plotting
#------------------------------------------------------------------------------

scale = 10.0174
distance = 0
path = np.array([(2850, 6400, mars_map[start[0], start[1]])]) # Coordenadas xyz iniciales
for i, (action, state) in enumerate(result.path()):
    if action == (0,1) or action == (1,0) or action == (0,-1) or action == (-1,0):
            distance += scale
    else:
        distance += (scale**2 + scale**2)**(1/2)
    path = np.concatenate([path, [(state[1]*scale, (nr-state[0])*scale, mars_map[state[0],state[1]])]])
print("A-star distance traveled: ", distance, '\n')
my_map.show(path)


