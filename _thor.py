import ai2thor.controller
import sys
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
 
 
class Form(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi("test5.ui", self)
        self.ui.show()
        self.CT = ai2thor.controller.Controller()
        self.CT.start(player_screen_width=1100,player_screen_height=700)
        self.event = self.CT.step(dict(action='Initialize', gridSize=0.25))
        self.scene_list = self.CT.scene_names()
        self.scene_name = self.scene_list[0]
        self.grid_size = 0.25
        self.teleport_x = -1.25
        self.teleport_y = 1.0
        self.teleport_z = -1.5

    @pyqtSlot()
    def start(self): 
        self.scene_name = self.ui.scene.text()
        self.grid_size = float(self.ui.GridSize.text())
        self.teleport_x = float(self.ui.x.text())
        self.teleport_y = float(self.ui.y.text())
        self.teleport_z = float(self.ui.z.text())

   
        print("scene : ", self.scene_name, ", grid_size : ", self.grid_size,", teleport : ",self.teleport_x,self.teleport_y,self.teleport_z)

        self.CT.reset(self.scene_name)
        self.event = self.CT.step(dict(action='Initialize', gridSize= self.grid_size))
        self.event = self.CT.step(dict(action='Teleport', x = self.teleport_x, y = self.teleport_y, z = self.teleport_z))
        
        

    @pyqtSlot()
    def reset(self): 
        self.scene_name = self.scene_list[0]
        self.grid_size = 0.25
        self.teleport_x = 0
        self.teleport_y = 0
        self.teleport_z = 0

        self.CT.reset(self.scene_name)
        self.event = self.CT.step(dict(action='Initialize', gridSize= self.grid_size))
        self.event = self.CT.step(dict(action='Teleport', x = self.teleport_x, y = self.teleport_y, z = self.teleport_z))

    # MoveAhead를 repeat만큼 반복
    @pyqtSlot()
    def go(self):
        self.event = self.CT.step(dict(action='MoveAhead'))
        
        
    # MoveBack을 repeat만큼 반복
    @pyqtSlot()
    def back(self):
        self.event = self.CT.step(dict(action='MoveBack'))

    # MoveRight을 repeat만큼 반복
    @pyqtSlot()
    def go_right(self):
        self.event = self.CT.step(dict(action='MoveRight'))
        
    # MoveLeft을 repeat만큼 반복
    @pyqtSlot()
    def go_left(self):
        self.event = self.CT.step(dict(action='MoveLeft'))


    def get_visible_objects(self):
        # output visible objects
        vis_objs = []
        for obj in self.event.metadata['objects']:
            if obj['visible']:
                vis_objs.append(obj)
        self.vis_objs = vis_objs
        return vis_objs

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Form()
    sys.exit(app.exec())
