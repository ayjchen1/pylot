"""This module implements an operator for visualizing the state of
the vehicle

IMPORTANT: modified to accomodate multiple cameras and obstacle detection +
error evaluation for each camera. other pipeline components have been
temporarily omitted for sake of simplicity."""

from collections import deque
from functools import partial

import erdos

import numpy as np

import pygame
from pygame.locals import K_n

import pylot.utils
from pylot.drivers.sensor_setup import RGBCameraSetup
from pylot.perception.camera_frame import CameraFrame
from pylot.planning.world import World

DEFAULT_VIS_TIME = 30000.0


class VisualizerOperator(erdos.Operator):
    """ The `VisualizerOperator` allows developers to see the current state
    of the entire pipeline by visualizing it on a pygame instance.

    This receives input data from almost the entire pipeline and renders the
    results of the operator currently chosen by the developer on the screen.
    
    TODO: currently, b/c of the 15 stream limit, limit this to obstacle
    error stream visuals only -> eventually, merge with visualizer_operator_old.py
    to include all streams
    """
    def __init__(self, pose_stream, front_camera_stream, back_camera_stream, tl_camera_stream,
                 prediction_camera_stream, depth_camera_stream,
                 point_cloud_stream, segmentation_stream, imu_stream,
                 obstacles_stream, front_obstacles_error, back_obstacles_error, traffic_lights_stream,
                 tracked_obstacles_stream, lane_detection_stream,
                 prediction_stream, waypoints_stream, control_stream,
                 display_control_stream, pygame_display, flags):
        visualize_streams = []
        self._pose_msgs = deque()
        pose_stream.add_callback(
            partial(self.save, msg_type="Pose", queue=self._pose_msgs))
        visualize_streams.append(pose_stream)

        rgb_camera_streams = [front_camera_stream, back_camera_stream]
        self._bgr_msg_queues = []
        for i in range(len(rgb_camera_streams)):
            self._bgr_msg_queues.append(deque())

            rgb_camera_stream = rgb_camera_streams[i]
            msg_name = "RGB" + str(i)
            rgb_camera_stream.add_callback(
                partial(self.save, msg_type=msg_name, queue=self._bgr_msg_queues[i]))
            visualize_streams.append(rgb_camera_stream)

        obstacles_error_streams = [front_obstacles_error, back_obstacles_error]
        self._obstacle_error_msg_queues = []
        for i in range(len(obstacles_error_streams)):
            self._obstacle_error_msg_queues.append(deque())

            obstacles_error_stream = obstacles_error_streams[i]
            msg_name = "ObstacleError" + str(i)
            obstacles_error_stream.add_callback(
                partial(self.save, msg_type=msg_name, queue=self._obstacle_error_msg_queues[i]))
            visualize_streams.append(obstacles_error_stream)

        self._control_msgs = deque()
        control_stream.add_callback(
            partial(self.save, msg_type="Control", queue=self._control_msgs))
        visualize_streams.append(control_stream)

        # Register a watermark callback on all the streams to be visualized.
        erdos.add_watermark_callback(visualize_streams, [], self.on_watermark)

        # Add a callback on a control stream to figure out what to display.
        display_control_stream.add_callback(self.change_display)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self.display = pygame_display

        # Set the font.
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self.font = pygame.font.Font(mono, 14)

        # Array of keys to figure out which message to display.
        self.current_display = 0
        self.display_array = []
        self.window_titles = []
        if flags.visualize_detected_obstacles: # eventually refactor + combine w/ other flags
            for i in range(len(obstacles_error_streams)):
                display_name = "ObstacleError" + str(i)
                self.display_array.append(display_name)
                self.window_titles.append("Detected obstacles")

        if flags.visualize_world:
            self._planning_world = World(flags, self._logger)
            top_down_transform = pylot.utils.get_top_down_transform(
                pylot.utils.Transform(pylot.utils.Location(),
                                      pylot.utils.Rotation()),
                flags.top_down_camera_altitude)
            self._bird_eye_camera_setup = RGBCameraSetup(
                'bird_eye_camera', flags.camera_image_width,
                flags.camera_image_height, top_down_transform, 90)
            self.display_array.append("PlanningWorld")
            self.window_titles.append("Planning world")
        else:
            self._planning_world = None

        assert len(self.display_array) == len(self.window_titles), \
            "The display and titles differ."
            
        # Save the flags.
        self._flags = flags

    @staticmethod
    def connect(pose_stream, front_camera_stream, back_camera_stream, tl_camera_stream,
                prediction_camera_stream, depth_stream, point_cloud_stream,
                segmentation_stream, imu_stream, obstacles_stream,
                front_obstacles_error, back_obstacles_error, 
                traffic_lights_stream, tracked_obstacles_stream,
                lane_detection_stream, prediction_stream, waypoints_stream,
                control_stream, display_control_stream):
        return []

    def save(self, msg, msg_type, queue):
        self._logger.debug("@{}: Received {} message.".format(
            msg.timestamp, msg_type))
        queue.append(msg)

    def change_display(self, display_message):
        if display_message.data == K_n:
            self.current_display = (self.current_display + 1) % len(
                self.display_array)
            self._logger.debug("@{}: Visualizer changed to {}".format(
                display_message.timestamp, self.current_display))

    def get_message(self, queue, timestamp, name):
        msg = None
        if queue:
            while len(queue) > 0:
                retrieved_msg = queue.popleft()
                if retrieved_msg.timestamp == timestamp:
                    msg = retrieved_msg
                    break
            if not msg:
                self._logger.warning(
                    "@{}: message for {} was not found".format(
                        timestamp, name))
        return msg

    def render_text(self, pose, control, timestamp):
        # Generate the text to be shown on the box.
        info_text = [
            "Display  : {}".format(self.window_titles[self.current_display]),
            "Timestamp: {}".format(timestamp.coordinates[0]),
        ]

        # Add information from the pose.
        if pose:
            info_text += [
                "Location : {:.1f}, {:.1f}, {:.1f}".format(
                    *tuple(pose.transform.location.as_numpy_array())),
                "Rotation : {:.1f}, {:.1f}, {:.1f}".format(
                    *tuple(pose.transform.rotation.as_numpy_array())),
                "Speed    : {:.2f} m/s".format(pose.forward_speed),
            ]

        # Add information from the control message
        if control:
            info_text += [
                "Throttle : {:.2f}".format(control.throttle),
                "Steer    : {:.2f}".format(control.steer),
                "Brake    : {:.2f}".format(control.brake),
                "Reverse  : {:.2f}".format(control.reverse),
            ]

        # Display the information box.
        info_surface = pygame.Surface(
            (220, self._flags.camera_image_height // 3))
        info_surface.set_alpha(100)
        self.display.blit(info_surface, (0, 0))

        # Render the text.
        v_offset = 10
        for line in info_text:
            if v_offset + 18 > self._flags.camera_image_height:
                break
            surface = self.font.render(line, True, (255, 255, 255))
            self.display.blit(surface, (8, v_offset))
            v_offset += 18
        pygame.display.flip()

    def on_watermark(self, timestamp):
        self._logger.debug("@{}: received watermark.".format(timestamp))

        pose_msg = self.get_message(self._pose_msgs, timestamp, "Pose")

        bgr_msgs = []
        for i in range(len(self._bgr_msg_queues)):
            msg_name = "BGR" + str(i)
            bgr_msg = self.get_message(self._bgr_msg_queues[i], timestamp, msg_name)
            bgr_msgs.append(bgr_msg)
        
        obstacle_error_msgs = []
        for i in range(len(self._obstacle_error_msg_queues)):
            msg_name = "ObstacleError" + str(i)
            obstacle_error_msg = self.get_message(self._obstacle_error_msg_queues[i], timestamp, msg_name)
            obstacle_error_msgs.append(obstacle_error_msg)
        
        control_msg = self.get_message(self._control_msgs, timestamp,
                                       "Control")
        if pose_msg:
            ego_transform = pose_msg.data.transform
        else:
            ego_transform = None

        # Add the visualizations on world.
        if self._flags.visualize_pose:
            self._visualize_pose(ego_transform)

        sensor_to_display = self.display_array[self.current_display]
        for i in range(len(self.display_array)):
            sensor_name = "ObstacleError" + str(i)
            bgr_msg = bgr_msgs[i]
            obstacle_error_msg = obstacle_error_msgs[i]

            if sensor_to_display == sensor_name and bgr_msg and obstacle_error_msg:
                bgr_msg.frame.annotate_with_bounding_boxes(timestamp,
                                                       obstacle_error_msg.obstacles,
                                                       ego_transform)
                bgr_msg.frame.visualize(self.display, timestamp=timestamp)
                break
    
        self.render_text(pose_msg.data, control_msg, timestamp)

    def run(self):
        # Run method is invoked after all operators finished initializing.
        # Thus, we're sure the world is up-to-date here.
        if (self._flags.visualize_pose or self._flags.visualize_imu
                or (self._flags.visualize_waypoints
                    and self._flags.draw_waypoints_on_world)):
            from pylot.simulation.utils import get_world
            _, self._world = get_world(self._flags.simulator_host,
                                       self._flags.simulator_port,
                                       self._flags.simulator_timeout)

    def _visualize_pose(self, ego_transform):
        # Draw position. We add 0.5 to z to ensure that the point is above
        # the road surface.
        loc = (ego_transform.location +
               pylot.utils.Location(0, 0, 0.5)).as_simulator_location()
        self._world.debug.draw_point(loc, size=0.2, life_time=DEFAULT_VIS_TIME)

    def _visualize_imu(self, msg):
        transform = msg.transform
        # Acceleration measured in ego frame, not global
        # z acceleration not useful for visualization so set to 0
        rotation_transform = pylot.utils.Transform(
            location=pylot.utils.Location(0, 0, 0),
            rotation=transform.rotation)
        rotated_acceleration = rotation_transform.transform_locations(
            [pylot.utils.Location(msg.acceleration.x, msg.acceleration.y,
                                  0)])[0]

        # Construct arrow.
        begin_acc = transform.location + pylot.utils.Location(z=0.5)
        end_acc = begin_acc + pylot.utils.Location(rotated_acceleration.x,
                                                   rotated_acceleration.y, 0)

        # draw arrow
        self._logger.debug("Acc: {}".format(rotated_acceleration))
        self._world.debug.draw_arrow(begin_acc.as_simulator_location(),
                                     end_acc.as_simulator_location(),
                                     arrow_size=0.1,
                                     life_time=0.1)
