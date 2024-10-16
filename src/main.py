import pygame as pg
import moderngl as mgl
import sys
from model import *

class GraphicsEngine:
    def __init__(self, win_size=(1600, 900)):
        pg.init()
        self.WIN_SIZE = win_size
        
        # set opengl attr
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)

        # create opengl context
        pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL | pg.DOUBLEBUF)

        # detect and use exiting opengl context
        self.ctx = mgl.create_context()

        # create clock
        self.clock = pg.time.Clock()

        # scene from model module
        self.scene = Cube(self)

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.scene.destroy()
                pg.quit()
                sys.exit()
    
    def render(self):
        self.ctx.clear(color=(0, 0, 0))
        self.scene.render()
        pg.display.flip()
    
    def run(self):
        while True:
            self.check_events()
            self.render()
            self.clock.tick(60)


if __name__ == "__main__":
    app = GraphicsEngine()
    app.run()
