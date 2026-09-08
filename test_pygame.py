import pygame
import sys

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Pygame Test Window")
    
    print("Pygame window should be open and RED now.")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        screen.fill((255, 0, 0)) # Red background
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
