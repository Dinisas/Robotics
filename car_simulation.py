"""
Robotics Lab Assignment 1 - Task 1
Lane Tracing Assist (LTA) Simulation - Joystick Control

Car model based on Ackermann steering kinematics from Lecture 6.
Controls:
- W: Accelerate
- A: Steer left
- D: Steer right
- S: Brake/Reverse
- ESC: Exit
"""

import pygame
import numpy as np
import sys

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 1400, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Lab 1 Task 1 - Car Simulation with Ackermann Steering")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Clock for controlling frame rate
clock = pygame.time.Clock()
FPS = 60

# Car parameters (Ackermann model from Lecture 6)
class Car:
    def __init__(self, x, y, theta):
        # Position and orientation
        self.x = x
        self.y = y
        self.theta = theta  # heading angle (radians)

        # Car dimensions
        self.length = 40  # car length (pixels)
        self.width = 20   # car width (pixels)
        self.wheelbase = 30  # L in the kinematic model (distance between axles)

        # Kinematic state
        self.velocity = 0.0  # V - linear velocity
        self.steering_angle = 0.0  # φ (phi) - steering angle (radians)

        # Control parameters
        self.max_velocity = 200.0  # pixels per second
        self.max_steering_angle = np.radians(35)  # maximum steering angle (35 degrees)
        self.acceleration = 100.0  # acceleration rate
        self.deceleration = 150.0  # deceleration/braking rate
        self.steering_rate = np.radians(60)  # steering change rate (degrees/second)

        # Friction
        self.friction = 50.0

    def update(self, dt, keys):
        """
        Update car state based on Ackermann steering model
        From Lecture 6 - Alternative formulation:
        ẋ = V * cos(θ)
        ẏ = V * sin(θ)
        θ̇ = V * tan(φ) / i
        φ̇ = ωs
        """

        # Handle acceleration input (W key)
        if keys[pygame.K_w]:
            self.velocity += self.acceleration * dt
        # Handle braking/reverse (S key)
        elif keys[pygame.K_s]:
            self.velocity -= self.deceleration * dt
        else:
            # Apply friction when no input
            if self.velocity > 0:
                self.velocity -= self.friction * dt
                if self.velocity < 0:
                    self.velocity = 0
            elif self.velocity < 0:
                self.velocity += self.friction * dt
                if self.velocity > 0:
                    self.velocity = 0

        # Limit velocity
        self.velocity = np.clip(self.velocity, -self.max_velocity * 0.5, self.max_velocity)

        # Handle steering input (A and D keys)
        if keys[pygame.K_a]:
            self.steering_angle += self.steering_rate * dt
        elif keys[pygame.K_d]:
            self.steering_angle -= self.steering_rate * dt
        else:
            # Return steering to center
            if abs(self.steering_angle) > 0.01:
                self.steering_angle *= 0.9
            else:
                self.steering_angle = 0

        # Limit steering angle
        self.steering_angle = np.clip(self.steering_angle, -self.max_steering_angle, self.max_steering_angle)

        # Ackermann steering kinematics (from Lecture 6)
        # Only update position if car is moving
        if abs(self.velocity) > 0.1:
            # Calculate angular velocity: ω = V * tan(φ) / L
            omega = self.velocity * np.tan(self.steering_angle) / self.wheelbase

            # Update position and orientation
            self.x += self.velocity * np.cos(self.theta) * dt
            self.y += self.velocity * np.sin(self.theta) * dt
            self.theta += omega * dt

            # Keep theta in [-π, π]
            self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

    def get_corners(self):
        """Get the four corners of the car for rendering"""
        # Car rectangle in local coordinates
        corners = np.array([
            [-self.length/2, -self.width/2],
            [self.length/2, -self.width/2],
            [self.length/2, self.width/2],
            [-self.length/2, self.width/2]
        ])

        # Rotation matrix
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        rotation = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        # Rotate and translate corners
        rotated = corners @ rotation.T
        translated = rotated + np.array([self.x, self.y])

        return translated

    def draw(self, screen):
        """Draw the car - transparent with only axis and wheels visible"""
        # Draw the main axis (chassis line from rear to front)
        rear_x = self.x - (self.length/2) * np.cos(self.theta)
        rear_y = self.y - (self.length/2) * np.sin(self.theta)
        front_x = self.x + (self.length/2) * np.cos(self.theta)
        front_y = self.y + (self.length/2) * np.sin(self.theta)

        # Draw main chassis axis
        pygame.draw.line(screen, WHITE, (int(rear_x), int(rear_y)),
                        (int(front_x), int(front_y)), 3)

        # Draw center point
        pygame.draw.circle(screen, YELLOW, (int(self.x), int(self.y)), 4)

        # Draw rear axle (fixed wheels)
        rear_axle_x = self.x - (self.length/2 - 5) * np.cos(self.theta)
        rear_axle_y = self.y - (self.length/2 - 5) * np.sin(self.theta)

        # Rear wheels (perpendicular to car body)
        rear_wheel_angle = self.theta + np.pi/2
        rear_wheel_half_width = self.width / 2
        rear_left_x = rear_axle_x + rear_wheel_half_width * np.cos(rear_wheel_angle)
        rear_left_y = rear_axle_y + rear_wheel_half_width * np.sin(rear_wheel_angle)
        rear_right_x = rear_axle_x - rear_wheel_half_width * np.cos(rear_wheel_angle)
        rear_right_y = rear_axle_y - rear_wheel_half_width * np.sin(rear_wheel_angle)

        # Draw rear axle
        pygame.draw.line(screen, RED, (int(rear_left_x), int(rear_left_y)),
                        (int(rear_right_x), int(rear_right_y)), 2)

        # Draw rear wheels
        wheel_length = 8
        rear_left_end_x = rear_left_x + wheel_length * np.cos(self.theta)
        rear_left_end_y = rear_left_y + wheel_length * np.sin(self.theta)
        rear_right_end_x = rear_right_x + wheel_length * np.cos(self.theta)
        rear_right_end_y = rear_right_y + wheel_length * np.sin(self.theta)

        pygame.draw.line(screen, RED, (int(rear_left_x), int(rear_left_y)),
                        (int(rear_left_end_x), int(rear_left_end_y)), 4)
        pygame.draw.line(screen, RED, (int(rear_right_x), int(rear_right_y)),
                        (int(rear_right_end_x), int(rear_right_end_y)), 4)

        # Draw front axle (steerable wheels)
        front_axle_x = self.x + (self.length/2 - 5) * np.cos(self.theta)
        front_axle_y = self.y + (self.length/2 - 5) * np.sin(self.theta)

        # Front wheels (at steering angle)
        front_wheel_angle = self.theta + np.pi/2
        front_wheel_half_width = self.width / 2
        front_left_x = front_axle_x + front_wheel_half_width * np.cos(front_wheel_angle)
        front_left_y = front_axle_y + front_wheel_half_width * np.sin(front_wheel_angle)
        front_right_x = front_axle_x - front_wheel_half_width * np.cos(front_wheel_angle)
        front_right_y = front_axle_y - front_wheel_half_width * np.sin(front_wheel_angle)

        # Draw front axle
        pygame.draw.line(screen, GREEN, (int(front_left_x), int(front_left_y)),
                        (int(front_right_x), int(front_right_y)), 2)

        # Draw steerable front wheels
        wheel_steering_angle = self.theta + self.steering_angle
        front_left_end_x = front_left_x + wheel_length * np.cos(wheel_steering_angle)
        front_left_end_y = front_left_y + wheel_length * np.sin(wheel_steering_angle)
        front_right_end_x = front_right_x + wheel_length * np.cos(wheel_steering_angle)
        front_right_end_y = front_right_y + wheel_length * np.sin(wheel_steering_angle)

        pygame.draw.line(screen, GREEN, (int(front_left_x), int(front_left_y)),
                        (int(front_left_end_x), int(front_left_end_y)), 4)
        pygame.draw.line(screen, GREEN, (int(front_right_x), int(front_right_y)),
                        (int(front_right_end_x), int(front_right_end_y)), 4)

        # Draw direction arrow at the front
        pygame.draw.circle(screen, BLUE, (int(front_x), int(front_y)), 5)


class SaoPauloTrack:
    """São Paulo F1 Circuit (Interlagos-inspired) with 2 lanes"""
    def __init__(self, offset_x=100, offset_y=100):
        self.offset_x = offset_x
        self.offset_y = offset_y

        # Lane configuration
        self.lane_width = 50
        self.track_width = 2 * self.lane_width  # 2 lanes

        # Define the track centerline points (approximating the circuit from the image)
        # Points are (x, y) relative to offset
        scale = 1.0
        self.centerline = [
            (800, 600),   # 1 - Start/Finish straight
            (750, 500),   # 2 - Pit exit merge
            (650, 400),   # 3 - Turn 1 (Descida do Lago)
            (550, 350),   # 4 -
            (450, 330),   # 5 - Turn 2 (Curva do Sol)
            (350, 300),   # 6 - Esses section
            (250, 250),   # 7 - Top of the hill
            (200, 180),   # 8 -
            (180, 120),   # 9 - Mergulho (downhill section)
            (200, 60),    # 10 -
            (300, 30),    # 11 - Flat-out back straight begins
            (500, 30),    # Continue straight
            (700, 30),    # Continue straight
            (900, 30),    # Continue straight
            (1100, 50),   # 11 - Turn approaching (Subida dos Boxes)
            (1150, 100),  # Start climbing
            (1180, 200),  # Climbing
            (1180, 300),  # Continue climb
            (1180, 400),  # Junção
            (1150, 500),  # Final corner
            (1100, 550),  # Entering main straight
            (1000, 600),  # Pit entry area
            (900, 600),   # Main straight
            (800, 600),   # Back to start
        ]

        # Adjust all points by offset and scale
        self.centerline = [(x * scale + offset_x, y * scale + offset_y)
                          for x, y in self.centerline]

    def draw(self, screen):
        """Draw the São Paulo-style track with 2 lanes"""
        # Draw outer boundary
        outer_points = self._offset_line(self.centerline, self.track_width / 2)
        if len(outer_points) > 2:
            pygame.draw.lines(screen, WHITE, True, outer_points, 4)

        # Draw inner boundary
        inner_points = self._offset_line(self.centerline, -self.track_width / 2)
        if len(inner_points) > 2:
            pygame.draw.lines(screen, WHITE, True, inner_points, 4)

        # Draw middle lane divider (dashed)
        self._draw_dashed_line(screen, self.centerline, WHITE, 2, 20, 15)

        # Draw start/finish line
        start_idx = 0
        if start_idx < len(self.centerline) - 1:
            p1 = self.centerline[start_idx]
            p2 = self.centerline[start_idx + 1]

            # Perpendicular line for start/finish
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                perp_x = -dy / length
                perp_y = dx / length

                start_left = (p1[0] + perp_x * self.track_width / 2,
                            p1[1] + perp_y * self.track_width / 2)
                start_right = (p1[0] - perp_x * self.track_width / 2,
                             p1[1] - perp_y * self.track_width / 2)

                # Draw checkered pattern
                num_checks = 10
                for i in range(num_checks):
                    t1 = i / num_checks
                    t2 = (i + 1) / num_checks
                    x1 = start_left[0] + t1 * (start_right[0] - start_left[0])
                    y1 = start_left[1] + t1 * (start_right[1] - start_left[1])
                    x2 = start_left[0] + t2 * (start_right[0] - start_left[0])
                    y2 = start_left[1] + t2 * (start_right[1] - start_left[1])

                    color = WHITE if i % 2 == 0 else BLACK
                    pygame.draw.line(screen, color, (int(x1), int(y1)),
                                   (int(x2), int(y2)), 5)

    def _offset_line(self, points, offset):
        """Offset a line perpendicular to its direction"""
        offset_points = []

        for i in range(len(points)):
            p_prev = points[i - 1] if i > 0 else points[-1]
            p_curr = points[i]
            p_next = points[(i + 1) % len(points)]

            # Calculate perpendicular direction
            dx1 = p_curr[0] - p_prev[0]
            dy1 = p_curr[1] - p_prev[1]
            len1 = np.sqrt(dx1**2 + dy1**2) or 1

            dx2 = p_next[0] - p_curr[0]
            dy2 = p_next[1] - p_curr[1]
            len2 = np.sqrt(dx2**2 + dy2**2) or 1

            # Average perpendicular direction
            perp_x = -(dy1/len1 + dy2/len2) / 2
            perp_y = (dx1/len1 + dx2/len2) / 2
            perp_len = np.sqrt(perp_x**2 + perp_y**2) or 1

            # Normalize and apply offset
            offset_x = p_curr[0] + (perp_x / perp_len) * offset
            offset_y = p_curr[1] + (perp_y / perp_len) * offset

            offset_points.append((int(offset_x), int(offset_y)))

        return offset_points

    def _draw_dashed_line(self, screen, points, color, width, dash_length, gap_length):
        """Draw a dashed line through the given points"""
        total_length = 0
        segments = []

        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            seg_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            segments.append((p1, p2, seg_length))
            total_length += seg_length

        dash_pattern = dash_length + gap_length
        current_length = 0

        for p1, p2, seg_length in segments:
            if seg_length == 0:
                continue

            dx = (p2[0] - p1[0]) / seg_length
            dy = (p2[1] - p1[1]) / seg_length

            seg_start = 0
            while seg_start < seg_length:
                # Determine if we're in a dash or gap
                pattern_pos = (current_length + seg_start) % dash_pattern

                if pattern_pos < dash_length:
                    # Draw dash
                    dash_start = seg_start
                    dash_end = min(seg_start + (dash_length - pattern_pos), seg_length)

                    x1 = p1[0] + dx * dash_start
                    y1 = p1[1] + dy * dash_start
                    x2 = p1[0] + dx * dash_end
                    y2 = p1[1] + dy * dash_end

                    pygame.draw.line(screen, color, (int(x1), int(y1)),
                                   (int(x2), int(y2)), width)

                    seg_start = dash_end
                else:
                    # Skip gap
                    seg_start += (dash_pattern - pattern_pos)

            current_length += seg_length

    def get_start_position(self, lane_number=1):
        """Get starting position in the specified lane (1=inner, 2=outer)"""
        # Start at the first point on the main straight
        start_point = self.centerline[0]
        next_point = self.centerline[1]

        # Calculate direction
        dx = next_point[0] - start_point[0]
        dy = next_point[1] - start_point[1]
        theta = np.arctan2(dy, dx)

        # Offset to lane center
        perp_angle = theta + np.pi / 2
        if lane_number == 1:
            # Inner lane
            offset = -self.lane_width / 2
        else:
            # Outer lane
            offset = self.lane_width / 2

        x = start_point[0] + offset * np.cos(perp_angle)
        y = start_point[1] + offset * np.sin(perp_angle)

        return x, y, theta


def draw_info_panel(screen, car):
    """Draw information panel showing car state"""
    font = pygame.font.Font(None, 28)

    info_texts = [
        f"Velocity: {car.velocity:.1f} px/s",
        f"Steering Angle: {np.degrees(car.steering_angle):.1f}°",
        f"Heading: {np.degrees(car.theta):.1f}°",
        f"Position: ({car.x:.0f}, {car.y:.0f})",
        "",
        "Controls:",
        "W - Accelerate",
        "S - Brake",
        "A - Steer Left",
        "D - Steer Right",
        "ESC - Exit"
    ]

    y_offset = 10
    for text in info_texts:
        surface = font.render(text, True, WHITE)
        screen.blit(surface, (10, y_offset))
        y_offset += 30


def main():
    # Create São Paulo-style track
    track = SaoPauloTrack(offset_x=50, offset_y=50)

    # Create car at the start position in the inner lane
    start_x, start_y, start_theta = track.get_start_position(lane_number=1)
    car = Car(start_x, start_y, start_theta)

    # Main game loop
    running = True
    dt = 1.0 / FPS

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Get keyboard state
        keys = pygame.key.get_pressed()

        # Update car
        car.update(dt, keys)

        # Draw everything
        screen.fill(BLACK)

        # Draw track
        track.draw(screen)

        # Draw car
        car.draw(screen)

        # Draw info panel
        draw_info_panel(screen, car)

        # Update display
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
