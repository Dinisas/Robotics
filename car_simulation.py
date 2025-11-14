"""
Robotics Lab Assignment 1
Task 1: Lane Tracing Assist (LTA) Simulation - Joystick Control
Task 2b: Camera Sensor Model for Lane Detection
Task 2c: Pure Pursuit Lane Keeping Assist (LKA) Controller

Car model based on Ackermann steering kinematics from Lecture 6.
Camera sensor models computer vision system similar to Comma AI setup.
LKA uses Pure Pursuit algorithm for autonomous lane keeping.

Controls:
- W: Accelerate
- S: Brake/Reverse
- A: Steer left (manual - deactivates LKA)
- D: Steer right (manual - deactivates LKA)
- F: Toggle LKA on/off
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
        self.max_velocity = 120.0  # pixels per second (reduced from 200)
        self.max_steering_angle = np.radians(35)  # maximum steering angle (35 degrees)
        self.acceleration = 50.0  # acceleration rate (reduced from 100)
        self.deceleration = 100.0  # deceleration/braking rate (reduced from 150)
        self.steering_rate = np.radians(60)  # steering change rate (degrees/second)

        # Friction
        self.friction = 30.0  # reduced from 50.0

    def update(self, dt, keys, lka_steering=None, lka_controller=None):
        """
        Update car state based on Ackermann steering model
        From Lecture 6 - Alternative formulation:
        ẋ = V * cos(θ)
        ẏ = V * sin(θ)
        θ̇ = V * tan(φ) / i
        φ̇ = ωs

        Args:
            dt: time step
            keys: keyboard state
            lka_steering: optional steering angle from LKA controller (radians)
            lka_controller: reference to LKA controller for override detection
        """

        # Handle acceleration input (W key) - LKA does NOT control speed
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

        # Handle steering: manual input overrides LKA
        manual_steering = keys[pygame.K_a] or keys[pygame.K_d]

        if manual_steering:
            # User is manually steering - deactivate LKA
            if lka_controller and lka_controller.active:
                lka_controller.deactivate()

            # Manual steering control
            if keys[pygame.K_a]:
                self.steering_angle -= self.steering_rate * dt  # Turn LEFT
            elif keys[pygame.K_d]:
                self.steering_angle += self.steering_rate * dt  # Turn RIGHT
        elif lka_steering is not None:
            # LKA is active and controlling steering
            self.steering_angle = lka_steering
        else:
            # No input - return steering to center
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

        # Draw rear wheeli
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

    def get_front_axle_position(self):
        """Return the world coordinates of the front axle center (used by camera/wheel calculations)"""
        front_axle_x = self.x + (self.length/2 - 5) * np.cos(self.theta)
        front_axle_y = self.y + (self.length/2 - 5) * np.sin(self.theta)
        return front_axle_x, front_axle_y

    def get_front_wheel_positions(self):
        """Return the world coordinates of the left and right front wheel centers"""
        # Front axle position
        front_axle_x = self.x + (self.length/2 - 5) * np.cos(self.theta)
        front_axle_y = self.y + (self.length/2 - 5) * np.sin(self.theta)

        # Perpendicular direction for wheel positions
        # theta + π/2 points to the left side of the car
        wheel_angle = self.theta + np.pi/2
        wheel_half_width = self.width / 2

        # Left wheel center (perpendicular left from car's perspective)
        left_wheel_x = front_axle_x + wheel_half_width * np.cos(wheel_angle)
        left_wheel_y = front_axle_y + wheel_half_width * np.sin(wheel_angle)

        # Right wheel center (perpendicular right from car's perspective)
        right_wheel_x = front_axle_x - wheel_half_width * np.cos(wheel_angle)
        right_wheel_y = front_axle_y - wheel_half_width * np.sin(wheel_angle)

        return (left_wheel_x, left_wheel_y), (right_wheel_x, right_wheel_y)

class CameraSensor:
    """
    Camera sensor model for lane detection (similar to Comma AI setup)
    Models a real forward-facing camera with computer vision capabilities
    """
    def __init__(self, car):
        self.car = car

        # Camera specifications (based on typical automotive cameras like Comma AI)
        self.field_of_view = np.radians(80)  # 80 degrees horizontal FOV (wider to see wheels)
        self.max_range = 300  # pixels (detection range)
        self.min_range = 20   # pixels (minimum detection distance)

        # Image resolution (typical automotive camera)
        self.image_width = 1280  # pixels
        self.image_height = 720  # pixels

        # Camera mounting position: behind the front axle (windshield position)
        # Needs to be far enough back to see the front wheels in its FOV
        self.mount_offset = self.car.length * 0.10  # Further back (10% from center)

        # Lane detection parameters
        self.detection_confidence = 0.95  # 95% detection confidence
        self.lane_sample_points = 10  # Number of points to sample along lane

        # Lane detection results
        self.left_lane_detected = False
        self.right_lane_detected = False
        self.left_lane_position = None  # (angle) relative to car (we do not expose distance)
        self.right_lane_position = None
        self.lane_center_offset = 0.0  # Lateral offset from lane center (radians)
        self.lane_heading_error = 0.0  # Angular error with lane direction
        self.current_lane = "UNKNOWN"  # Which lane the car is currently in

    def get_camera_position(self):
        """Get the world position of the camera"""
        camera_x = self.car.x + self.mount_offset * np.cos(self.car.theta)
        camera_y = self.car.y + self.mount_offset * np.sin(self.car.theta)
        return camera_x, camera_y

    def detect_lanes(self, track):
        """
        Detect lane lines within the camera's field of view
        The track has 3 lane boundaries: left outer, center (dotted), right outer
        Camera detects the two boundaries of whichever lane the car is currently in
        Returns: (left_lane_points, right_lane_points, center_lane_points)
        """
        camera_x, camera_y = self.get_camera_position()
        camera_angle = self.car.theta

        # Get all three track boundaries
        # Track has 2 lanes, so 3 boundaries: outer left, center dotted, outer right
        left_outer_boundary = track._offset_line(track.centerline, -track.lane_width)
        center_boundary = track.centerline  # Middle dotted line
        right_outer_boundary = track._offset_line(track.centerline, track.lane_width)

        # Detect all three boundaries
        left_outer_points = self._detect_lane_boundary(
            left_outer_boundary, camera_x, camera_y, camera_angle
        )
        center_points = self._detect_lane_boundary(
            center_boundary, camera_x, camera_y, camera_angle
        )
        right_outer_points = self._detect_lane_boundary(
            right_outer_boundary, camera_x, camera_y, camera_angle
        )

        # Determine which lane the car is in by checking lateral position
        # relative to track center
        car_lateral_offset = self._get_lateral_offset_from_track_center(track)

        # If car is in left lane (negative offset), show left outer and center
        # If car is in right lane (positive offset), show center and right outer
        if car_lateral_offset < 0:
            # Car is in LEFT lane
            left_lane_points = left_outer_points
            right_lane_points = center_points
            current_lane = "LEFT"
        else:
            # Car is in RIGHT lane
            left_lane_points = center_points
            right_lane_points = right_outer_points
            current_lane = "RIGHT"

        # Update detection status
        self.left_lane_detected = len(left_lane_points) > 0
        self.right_lane_detected = len(right_lane_points) > 0
        self.current_lane = current_lane  # Store which lane we're in

        # Calculate lane positions relative to car
        if self.left_lane_detected and len(left_lane_points) > 0:
            # store angle (no distance measurement exposed)
            self.left_lane_position = self._calculate_lane_position(left_lane_points[0])

        if self.right_lane_detected and len(right_lane_points) > 0:
            self.right_lane_position = self._calculate_lane_position(right_lane_points[0])

        # Calculate lane center offset and heading error
        self._calculate_lane_tracking_errors(left_lane_points, right_lane_points)

        return left_lane_points, right_lane_points, center_points

    def _detect_lane_boundary(self, boundary_points, camera_x, camera_y, camera_angle):
        """Detect lane boundary points visible to the camera"""
        visible_points = []

        for point in boundary_points:
            px, py = point

            # Vector from camera to point
            dx = px - camera_x
            dy = py - camera_y
            distance = np.sqrt(dx**2 + dy**2)

            # Check if within range (we still use range internally for visibility,
            # but we do NOT expose distance as a measurement in outputs)
            if distance < self.min_range or distance > self.max_range:
                continue

            # Angle to point relative to camera direction
            point_angle = np.arctan2(dy, dx)
            angle_diff = point_angle - camera_angle

            # Normalize angle to [-π, π]
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

            # Check if within field of view
            if abs(angle_diff) < self.field_of_view / 2:
                # Append (px, py, angle) - do NOT expose distance
                visible_points.append((px, py, angle_diff))

        return visible_points

    def _calculate_lane_position(self, point_data):
        """Calculate lane position relative to car (angle only). We intentionally
        do not return distance to simulate a camera that provides angular lane
        measurements / pixel angular position rather than absolute range."""
        px, py, angle = point_data
        return angle

    def _get_lateral_offset_from_track_center(self, track):
        """
        Calculate car's lateral offset from track centerline
        Negative = left of center, Positive = right of center
        """
        # Find closest point on track centerline
        min_dist = float('inf')
        closest_idx = 0

        for i, (cx, cy) in enumerate(track.centerline):
            dist = np.sqrt((self.car.x - cx)**2 + (self.car.y - cy)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Get the track direction at this point
        p_curr = track.centerline[closest_idx]
        p_next = track.centerline[(closest_idx + 1) % len(track.centerline)]

        # Calculate perpendicular direction (left side of track)
        dx = p_next[0] - p_curr[0]
        dy = p_next[1] - p_curr[1]
        track_angle = np.arctan2(dy, dx)

        # Vector from track center to car
        to_car_x = self.car.x - p_curr[0]
        to_car_y = self.car.y - p_curr[1]

        # Project onto perpendicular (left is positive in track-relative coords)
        perp_angle = track_angle + np.pi / 2
        lateral_offset = (to_car_x * np.cos(perp_angle) +
                         to_car_y * np.sin(perp_angle))

        return lateral_offset

    def _calculate_lane_tracking_errors(self, left_points, right_points):
        """Calculate lateral offset and heading error from lane center"""
        if not left_points or not right_points:
            return

        # Choose points that are most centered in the camera view (smallest abs(angle))
        left_closest = min(left_points, key=lambda p: abs(p[2]))
        right_closest = min(right_points, key=lambda p: abs(p[2]))

        # Extract angles
        left_angle = left_closest[2]
        right_angle = right_closest[2]

        # Lateral offset is represented as average angular offset (radians)
        self.lane_center_offset = (right_angle + left_angle) / 2

        # Heading error (simplified) uses same angular offset
        self.lane_heading_error = self.lane_center_offset

    def draw_camera_view(self, screen, track):
        """Visualize the camera's field of view and detected lanes"""
        camera_x, camera_y = self.get_camera_position()

        # Draw field of view cone
        fov_color = (0, 255, 0, 50)  # Semi-transparent green
        fov_points = [
            (camera_x, camera_y)
        ]

        # Left edge of FOV
        left_angle = self.car.theta - self.field_of_view / 2
        fov_points.append((
            camera_x + self.max_range * np.cos(left_angle),
            camera_y + self.max_range * np.sin(left_angle)
        ))

        # Right edge of FOV
        right_angle = self.car.theta + self.field_of_view / 2
        fov_points.append((
            camera_x + self.max_range * np.cos(right_angle),
            camera_y + self.max_range * np.sin(right_angle)
        ))

        # Draw FOV cone (semi-transparent)
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(s, fov_color, fov_points)
        screen.blit(s, (0, 0))

        # Draw FOV edges
        pygame.draw.line(screen, GREEN, (int(camera_x), int(camera_y)),
                        (int(fov_points[1][0]), int(fov_points[1][1])), 1)
        pygame.draw.line(screen, GREEN, (int(camera_x), int(camera_y)),
                        (int(fov_points[2][0]), int(fov_points[2][1])), 1)

        # Detect and draw lane lines
        left_lane, right_lane, center_lane = self.detect_lanes(track)

        # Draw detected lane points (multiple points along each boundary)
        # Draw vectors from LEFT wheel to left lane points, RIGHT wheel to right lane points
        # This simulates the car knowing where its wheels are relative to lane boundaries
        (left_wheel_x, left_wheel_y), (right_wheel_x, right_wheel_y) = self.car.get_front_wheel_positions()

        # Draw left lane points and vectors from LEFT WHEEL to left boundary
        for point in left_lane:
            px, py, ang = point
            pygame.draw.circle(screen, (255, 0, 0), (int(px), int(py)), 3)
            # Vector from left wheel to left lane point
            pygame.draw.line(screen, (255, 128, 0),
                           (int(left_wheel_x), int(left_wheel_y)),
                           (int(px), int(py)), 1)

        # Draw right lane points and vectors from RIGHT WHEEL to right boundary
        for point in right_lane:
            px, py, ang = point
            pygame.draw.circle(screen, (0, 128, 255), (int(px), int(py)), 3)
            # Vector from right wheel to right lane point
            pygame.draw.line(screen, (0, 200, 200),
                           (int(right_wheel_x), int(right_wheel_y)),
                           (int(px), int(py)), 1)

        # Also draw center boundary points (dotted middle line) in dark blue
        for point in center_lane:
            px, py, ang = point
            pygame.draw.circle(screen, (0, 0, 200), (int(px), int(py)), 2)

        # Draw camera position (green)
        pygame.draw.circle(screen, GREEN, (int(camera_x), int(camera_y)), 5)

        # Draw wheel positions for clarity with distinct colors
        # Left wheel = orange/red, Right wheel = cyan/blue
        pygame.draw.circle(screen, (255, 100, 0), (int(left_wheel_x), int(left_wheel_y)), 5)  # Orange for left wheel
        pygame.draw.circle(screen, (0, 150, 255), (int(right_wheel_x), int(right_wheel_y)), 5)  # Cyan for right wheel

    def draw_camera_data_panel(self, screen):
        """Draw a panel showing camera sensor data"""
        font = pygame.font.Font(None, 24)
        panel_x = WIDTH - 350
        panel_y = 10

        # Semi-transparent background
        s = pygame.Surface((340, 280), pygame.SRCALPHA)
        pygame.draw.rect(s, (0, 0, 0, 180), (0, 0, 340, 280))
        screen.blit(s, (panel_x, panel_y))

        # Camera sensor information
        info_texts = [
            "=== CAMERA SENSOR (Task 2b) ===",
            f"Field of View: {np.degrees(self.field_of_view):.1f}°",
            f"Detection Range: {self.min_range}-{self.max_range}px",
            f"Resolution: {self.image_width}x{self.image_height}",
            f"Confidence: {self.detection_confidence*100:.0f}%",
            "",
            "=== LANE DETECTION ===",
            f"Current Lane: {self.current_lane}",
            f"Left Boundary: {'DETECTED' if self.left_lane_detected else 'NOT DETECTED'}",
            f"Right Boundary: {'DETECTED' if self.right_lane_detected else 'NOT DETECTED'}",
            "",
            f"Lane Center Offset: {np.degrees(self.lane_center_offset):.2f}°",
            f"Heading Error: {np.degrees(self.lane_heading_error):.2f}°",
        ]

        y_offset = panel_y + 10
        for text in info_texts:
            if text.startswith("==="):
                color = YELLOW
            elif "DETECTED" in text:
                color = GREEN if "DETECTED" in text and "NOT" not in text else RED
            else:
                color = WHITE

            surface = font.render(text, True, color)
            screen.blit(surface, (panel_x + 10, y_offset))
            y_offset += 25


class PurePursuitLKA:
    """
    Pure Pursuit Lane Keeping Assist Controller
    Based on lecture material - follows lane center using look-ahead point
    """
    def __init__(self, car, camera):
        self.car = car
        self.camera = camera

        # LKA state
        self.active = False
        self.was_manually_overridden = False

        # Pure Pursuit parameters
        self.base_lookahead_distance = 80.0  # Base look-ahead distance in pixels
        self.lookahead_gain = 0.5  # Multiply by velocity for adaptive look-ahead
        self.min_lookahead = 40.0  # Minimum look-ahead distance
        self.max_lookahead = 150.0  # Maximum look-ahead distance

        # Control parameters
        self.steering_gain = 1.2  # Amplify steering response

    def toggle(self):
        """Toggle LKA on/off"""
        self.active = not self.active
        self.was_manually_overridden = False
        return self.active

    def deactivate(self):
        """Deactivate LKA (called when user manually steers)"""
        if self.active:
            self.active = False
            self.was_manually_overridden = True

    def calculate_steering(self, track):
        """
        Pure Pursuit algorithm to calculate steering angle
        Returns: steering_angle (radians) or None if LKA should not control
        """
        if not self.active:
            return None

        # Detect lanes
        left_lane, right_lane, center_lane = self.camera.detect_lanes(track)

        # Need both lane boundaries detected
        if not (self.camera.left_lane_detected and self.camera.right_lane_detected):
            return None

        # Calculate adaptive look-ahead distance based on speed
        speed = abs(self.car.velocity)
        lookahead_distance = self.base_lookahead_distance + self.lookahead_gain * speed
        lookahead_distance = np.clip(lookahead_distance, self.min_lookahead, self.max_lookahead)

        # Get car position and heading
        car_x = self.car.x
        car_y = self.car.y
        car_theta = self.car.theta

        # Calculate lane center points (average of left and right lane boundaries)
        lane_center_points = []

        # Match left and right points that are at similar distances
        for left_point in left_lane:
            left_x, left_y, left_ang = left_point
            # Find closest right point
            min_dist = float('inf')
            closest_right = None

            for right_point in right_lane:
                right_x, right_y, right_ang = right_point
                dist = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_right = right_point

            if closest_right:
                right_x, right_y, right_ang = closest_right
                # Calculate midpoint (lane center)
                center_x = (left_x + right_x) / 2
                center_y = (left_y + right_y) / 2

                # Distance from car to this center point
                dx = center_x - car_x
                dy = center_y - car_y
                distance = np.sqrt(dx**2 + dy**2)

                lane_center_points.append((center_x, center_y, distance))

        if len(lane_center_points) == 0:
            return None

        # Find the look-ahead point: closest point to the desired look-ahead distance
        best_point = min(lane_center_points,
                        key=lambda p: abs(p[2] - lookahead_distance))

        lookahead_x, lookahead_y, actual_distance = best_point

        # Calculate angle to look-ahead point (α in the formula)
        dx = lookahead_x - car_x
        dy = lookahead_y - car_y
        angle_to_point = np.arctan2(dy, dx)

        # Alpha: angle between car heading and look-ahead point
        alpha = angle_to_point - car_theta
        # Normalize to [-π, π]
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        # Pure Pursuit formula: δ = arctan(2 * L * sin(α) / ld)
        wheelbase = self.car.wheelbase

        if actual_distance < 1.0:  # Avoid division by zero
            return 0.0

        steering_angle = np.arctan2(2 * wheelbase * np.sin(alpha), actual_distance)

        # Apply gain and limit to car's max steering angle
        steering_angle *= self.steering_gain
        steering_angle = np.clip(steering_angle,
                                -self.car.max_steering_angle,
                                self.car.max_steering_angle)

        # Store look-ahead point for visualization
        self.lookahead_point = (lookahead_x, lookahead_y)
        self.lookahead_distance = actual_distance

        return steering_angle

    def draw_status(self, screen):
        """Draw LKA status indicator"""
        font = pygame.font.Font(None, 32)

        if self.active:
            status_text = "LKA: ACTIVE"
            color = GREEN
            bg_color = (0, 100, 0, 150)
        elif self.was_manually_overridden:
            status_text = "LKA: OVERRIDE"
            color = YELLOW
            bg_color = (100, 100, 0, 150)
        else:
            status_text = "LKA: OFF"
            color = RED
            bg_color = (100, 0, 0, 150)

        # Draw background box
        s = pygame.Surface((200, 50), pygame.SRCALPHA)
        pygame.draw.rect(s, bg_color, (0, 0, 200, 50))
        screen.blit(s, (WIDTH // 2 - 100, 10))

        # Draw text
        surface = font.render(status_text, True, color)
        text_rect = surface.get_rect(center=(WIDTH // 2, 35))
        screen.blit(surface, text_rect)

        # Draw F key hint
        hint_font = pygame.font.Font(None, 20)
        hint_text = "Press F to toggle"
        hint_surface = hint_font.render(hint_text, True, WHITE)
        hint_rect = hint_surface.get_rect(center=(WIDTH // 2, 55))
        screen.blit(hint_surface, hint_rect)

    def draw_lookahead_point(self, screen):
        """Draw the look-ahead point and path"""
        if self.active and hasattr(self, 'lookahead_point'):
            lx, ly = self.lookahead_point

            # Draw line from car to look-ahead point
            pygame.draw.line(screen, (255, 255, 0),
                           (int(self.car.x), int(self.car.y)),
                           (int(lx), int(ly)), 2)

            # Draw look-ahead point as a yellow circle
            pygame.draw.circle(screen, YELLOW, (int(lx), int(ly)), 8)
            pygame.draw.circle(screen, (255, 200, 0), (int(lx), int(ly)), 10, 2)

            # Draw look-ahead distance circle
            pygame.draw.circle(screen, (255, 255, 0, 50),
                             (int(self.car.x), int(self.car.y)),
                             int(self.lookahead_distance), 1)


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
        "A/D - Steer (disables LKA)",
        "F - Toggle LKA",
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

    # Create camera sensor (Task 2b)
    camera = CameraSensor(car)

    # Create Pure Pursuit LKA controller (Task 2c)
    lka = PurePursuitLKA(car, camera)

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
                elif event.key == pygame.K_f:
                    # Toggle LKA on/off
                    lka.toggle()

        # Get keyboard state
        keys = pygame.key.get_pressed()

        # Calculate LKA steering if active
        lka_steering = lka.calculate_steering(track) if lka.active else None

        # Update car (LKA can control steering, user controls speed)
        car.update(dt, keys, lka_steering, lka)

        # Draw everything
        screen.fill(BLACK)

        # Draw track
        track.draw(screen)

        # Draw camera field of view and detected lanes
        camera.draw_camera_view(screen, track)

        # Draw LKA look-ahead point if active
        lka.draw_lookahead_point(screen)

        # Draw car
        car.draw(screen)

        # Draw info panels
        draw_info_panel(screen, car)
        camera.draw_camera_data_panel(screen)

        # Draw LKA status
        lka.draw_status(screen)

        # Update display
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
