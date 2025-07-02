import pygame
import sys
import math

class SmartClimateVisualizer:
    def __init__(self, width=500, height=400):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('SmartClimate RL Visualization')
        self.font = pygame.font.SysFont('Arial', 20)
        self.clock = pygame.time.Clock()
        self.running = True

    def render(self, room_temp, num_people, ac_setting, light_states, outside_temp, time_of_day):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()
        self.screen.fill((230, 230, 230))
        # Comfort zone color
        if 20 <= room_temp <= 24:
            color = (120, 255, 120)
        elif 18 <= room_temp <= 26:
            color = (255, 255, 120)
        elif 16 <= room_temp <= 28:
            color = (255, 200, 120)
        else:
            color = (255, 120, 120)
        pygame.draw.rect(self.screen, color, (50, 50, 400, 200))
        # Room temp
        temp_text = self.font.render(f'Room Temp: {room_temp:.1f}C', True, (0,0,0))
        self.screen.blit(temp_text, (60, 60))
        # Occupancy
        occ_text = self.font.render(f'Occupancy: {num_people}', True, (0,0,0))
        self.screen.blit(occ_text, (60, 90))
        # AC setting
        ac_text = self.font.render(f'AC Setting: {ac_setting:.1f}C', True, (0,0,0))
        self.screen.blit(ac_text, (60, 120))
        # Outside temp
        out_text = self.font.render(f'Outside Temp: {outside_temp:.1f}C', True, (0,0,0))
        self.screen.blit(out_text, (60, 150))
        # Time
        hour = int(time_of_day)
        minute = int((time_of_day - hour) * 60)
        time_text = self.font.render(f'Time: {hour:02d}:{minute:02d}', True, (0,0,0))
        self.screen.blit(time_text, (60, 180))
        # Lights
        for i, state in enumerate(light_states):
            color = (255, 255, 0) if state else (180, 180, 180)
            pygame.draw.circle(self.screen, color, (350 + i*30, 100), 12)
        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        pygame.quit() 