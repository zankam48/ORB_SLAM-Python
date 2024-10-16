import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2
import numpy as np


def main():
    pygame.init()
    display = (640, 480)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    glViewport(0, 0, display[0], display[1])
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

    glEnable(GL_DEPTH_TEST)

    cap = cv2.VideoCapture(0)

    texture_id = glGenTextures(1)

    def draw_cube():
        glBegin(GL_QUADS)
        glColor3f(1.0, 0.0, 0.0)  
        glVertex3f(-1.0, -1.0,  1.0)
        glVertex3f( 1.0, -1.0,  1.0)
        glVertex3f( 1.0,  1.0,  1.0)
        glVertex3f(-1.0,  1.0,  1.0)

        glColor3f(0.0, 1.0, 0.0)  
        glVertex3f(-1.0, -1.0, -1.0)
        glVertex3f(-1.0,  1.0, -1.0)
        glVertex3f( 1.0,  1.0, -1.0)
        glVertex3f( 1.0, -1.0, -1.0)

        glColor3f(0.0, 0.0, 1.0)  
        glVertex3f(-1.0, -1.0, -1.0)
        glVertex3f(-1.0, -1.0,  1.0)
        glVertex3f(-1.0,  1.0,  1.0)
        glVertex3f(-1.0,  1.0, -1.0)

        glColor3f(1.0, 1.0, 0.0)  
        glVertex3f(1.0, -1.0, -1.0)
        glVertex3f(1.0,  1.0, -1.0)
        glVertex3f(1.0,  1.0,  1.0)
        glVertex3f(1.0, -1.0,  1.0)
        
        glColor3f(1.0, 0.0, 1.0)  
        glVertex3f(-1.0,  1.0, -1.0)
        glVertex3f(-1.0,  1.0,  1.0)
        glVertex3f( 1.0,  1.0,  1.0)
        glVertex3f( 1.0,  1.0, -1.0)
        
        glColor3f(0.0, 1.0, 1.0)  
        glVertex3f(-1.0, -1.0, -1.0)
        glVertex3f( 1.0, -1.0, -1.0)
        glVertex3f( 1.0, -1.0,  1.0)
        glVertex3f(-1.0, -1.0,  1.0)
        glEnd()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                return
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    cap.release()
                    pygame.quit()
                    return

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 0)

        frame_height, frame_width, _ = frame.shape

        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame_width, frame_height, 0, GL_RGB, GL_UNSIGNED_BYTE, frame)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glEnable(GL_TEXTURE_2D)
        glLoadIdentity()
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex3f(-4, -3, -5)
        glTexCoord2f(1, 0)
        glVertex3f(4, -3, -5)
        glTexCoord2f(1, 1)
        glVertex3f(4, 3, -5)
        glTexCoord2f(0, 1)
        glVertex3f(-4, 3, -5)
        glEnd()
        glDisable(GL_TEXTURE_2D)

        glLoadIdentity()
        glTranslatef(0.0, 0.0, -6)
        glRotatef(pygame.time.get_ticks() * 0.05, 1, 1, 0)
        draw_cube()

        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()
