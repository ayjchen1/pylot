import signal

from absl import app, flags

import erdos

import pylot.flags
import pylot.component_creator  # noqa: I100
import pylot.operator_creator
import pylot.utils
from pylot.drivers.sensor_setup import RGBCameraSetup
from pylot.simulation.utils import get_world, set_asynchronous_mode

FLAGS = flags.FLAGS

flags.DEFINE_list('goal_location', '234, 59, 39', 'Ego-vehicle goal location')

# The location of the center camera relative to the ego-vehicle.
FRONT_CAMERA_LOCATION = pylot.utils.Location(1.3, 0.0, 1.8) # x -> into the screen, y -> left/right, z -> up/down
BACK_CAMERA_LOCATION = pylot.utils.Location(-1.3, 0.0, 1.8)

def add_evaluation_operators(vehicle_id_stream, pose_stream, imu_stream,
                             pose_stream_for_control,
                             waypoints_stream_for_control):
    if FLAGS.evaluation:
        # Add the collision sensor.
        collision_stream = pylot.operator_creator.add_collision_sensor(
            vehicle_id_stream)

        # Add the lane invasion sensor.
        lane_invasion_stream = pylot.operator_creator.add_lane_invasion_sensor(
            vehicle_id_stream)

        # Add the traffic light invasion sensor.
        traffic_light_invasion_stream = \
            pylot.operator_creator.add_traffic_light_invasion_sensor(
                vehicle_id_stream, pose_stream)

        # Add the evaluation logger.
        pylot.operator_creator.add_eval_metric_logging(
            collision_stream, lane_invasion_stream,
            traffic_light_invasion_stream, imu_stream, pose_stream)

        # Add control evaluation logging operator.
        pylot.operator_creator.add_control_evaluation(
            pose_stream_for_control, waypoints_stream_for_control)

def add_cameras(vehicle_id_stream, release_sensor_stream, notify_streams, transforms):
    """
    Adds 1+ cameras to the vehicle
    """
    rgb_camera_streams = []
    rgb_camera_setups = []
    depth_camera_streams = []
    segmented_camera_streams = []
    point_cloud_streams = []
    lidar_setups = []
    depth_streams = []

    for transform in transforms:
        (cur_camera_stream, notify_rgb_stream,
         cur_camera_setup) = pylot.operator_creator.add_rgb_camera(
             transform, vehicle_id_stream, release_sensor_stream)
        notify_streams.append(notify_rgb_stream)
        rgb_camera_streams.append(cur_camera_stream)
        rgb_camera_setups.append(cur_camera_setup)

        if pylot.flags.must_add_depth_camera_sensor():
            (depth_camera_stream, notify_depth_stream,
             depth_camera_setup) = pylot.operator_creator.add_depth_camera(
                 transform, vehicle_id_stream, release_sensor_stream)
        else:
            depth_camera_stream = None
        depth_camera_streams.append(depth_camera_stream)

        if pylot.flags.must_add_segmented_camera_sensor():
            (ground_segmented_stream, notify_segmented_stream,
             _) = pylot.operator_creator.add_segmented_camera(
                 transform, vehicle_id_stream, release_sensor_stream)
        else:
            ground_segmented_stream = None
        segmented_camera_streams.append(ground_segmented_stream)

        if pylot.flags.must_add_lidar_sensor():
            # Place LiDAR sensor in the same location as the center camera.
            (point_cloud_stream, notify_lidar_stream,
             lidar_setup) = pylot.operator_creator.add_lidar(
                 transform, vehicle_id_stream, release_sensor_stream)
        else:
            point_cloud_stream = None
            lidar_setup = None
        point_cloud_streams.append(point_cloud_stream)
        lidar_setups.append(lidar_setup)

        if FLAGS.obstacle_location_finder_sensor == 'lidar':
            depth_stream = point_cloud_stream
            # Camera sensors are slower than the lidar sensor.
            notify_streams.append(notify_lidar_stream)
        elif FLAGS.obstacle_location_finder_sensor == 'depth_camera':
            depth_stream = depth_camera_stream
            notify_streams.append(notify_depth_stream)
        else:
            raise ValueError(
                'Unknown --obstacle_location_finder_sensor value {}'.format(
                    FLAGS.obstacle_location_finder_sensor))
        depth_streams.append(depth_stream)

    return rgb_camera_streams, rgb_camera_setups, depth_camera_streams, segmented_camera_streams, point_cloud_streams, lidar_setups, depth_streams

def driver():
    csv_logger = erdos.utils.setup_csv_logging("head-csv", FLAGS.csv_log_file_name)
    csv_logger.info('TIME,SIMTIME,CAMERA,GROUNDID,LABEL,X,Y,Z,ERROR')

    streams_to_send_top_on = []
    control_loop_stream = erdos.LoopStream()
    time_to_decision_loop_stream = erdos.LoopStream()
    if FLAGS.simulator_mode == 'pseudo-asynchronous':
        release_sensor_stream = erdos.LoopStream()
        pipeline_finish_notify_stream = erdos.LoopStream()
    else:
        release_sensor_stream = erdos.IngestStream()
        pipeline_finish_notify_stream = erdos.IngestStream()
    notify_streams = []

    # Create operator that bridges between pipeline and the simulator.
    (
        pose_stream,
        pose_stream_for_control,
        ground_traffic_lights_stream,
        ground_obstacles_stream,
        ground_speed_limit_signs_stream,
        ground_stop_signs_stream,
        vehicle_id_stream,
        open_drive_stream,
        global_trajectory_stream,
    ) = pylot.operator_creator.add_simulator_bridge(
        control_loop_stream,
        release_sensor_stream,
        pipeline_finish_notify_stream,
    )

    # Add sensors.
    front_transform = pylot.utils.Transform(FRONT_CAMERA_LOCATION, pylot.utils.Rotation(pitch=-5))
    back_transform = pylot.utils.Transform(BACK_CAMERA_LOCATION, pylot.utils.Rotation(pitch=0, yaw=180)) # CORRECT ROTATION?

    #camera_names = ["FRONT", "BACK"]
    #transforms = [front_transform, back_transform]
    camera_names = ["FRONT"]
    transforms = [front_transform]
    rgb_camera_streams, rgb_camera_setups, depth_camera_streams, segmented_camera_streams, point_cloud_streams, lidar_setups, depth_streams = \
            add_cameras(vehicle_id_stream, release_sensor_stream, notify_streams, transforms)

    # add a birds-eye camera

    top_down_transform = pylot.utils.get_top_down_transform(
                pylot.utils.Transform(pylot.utils.Location(),
                                      pylot.utils.Rotation()),
                                      FLAGS.top_down_camera_altitude)
    (bird_camera_stream, notify_bird_stream, bird_setup) = \
        pylot.operator_creator.add_bird_camera(top_down_transform, vehicle_id_stream, release_sensor_stream)
    notify_streams.append(notify_bird_stream)

    imu_stream = None
    if pylot.flags.must_add_imu_sensor():
        (imu_stream, _) = pylot.operator_creator.add_imu(
            pylot.utils.Transform(location=pylot.utils.Location(),
                                  rotation=pylot.utils.Rotation()),
            vehicle_id_stream)

    gnss_stream = None
    if pylot.flags.must_add_gnss_sensor():
        (gnss_stream, _) = pylot.operator_creator.add_gnss(
            pylot.utils.Transform(location=pylot.utils.Location(),
                                  rotation=pylot.utils.Rotation()),
            vehicle_id_stream)

# ----------------------------------------------------------------------------------

    #IGNORE
    if FLAGS.localization:
        pose_stream = pylot.operator_creator.add_localization(
            imu_stream, gnss_stream, pose_stream)

    # for each camera
    obstacles_streams = []
    perfect_obstacles_streams = []
    obstacles_error_streams = []
    for i in range(len(transforms)):
        transform = transforms[i]
        rgb_camera_stream = rgb_camera_streams[i]
        rgb_camera_setup = rgb_camera_setups[i]
        depth_camera_stream = depth_camera_streams[i]
        ground_segmented_stream = segmented_camera_streams[i]
        point_cloud_stream = point_cloud_streams[i]
        lidar_setup = lidar_setups[i]
        depth_stream = depth_streams[i]
        eval_name = camera_names[i] + "-eval"

        obstacles_stream, perfect_obstacles_stream, obstacles_error_stream = \
            pylot.component_creator.add_obstacle_detection(
                rgb_camera_stream, rgb_camera_setup, pose_stream,
                depth_stream, depth_camera_stream, ground_segmented_stream,
                ground_obstacles_stream, ground_speed_limit_signs_stream,
                ground_stop_signs_stream, time_to_decision_loop_stream, eval_name)

        print("CREATED COMPONENTS: ", transform.location)

        obstacles_streams.append(obstacles_stream)
        perfect_obstacles_streams.append(perfect_obstacles_stream)
        obstacles_error_streams.append(obstacles_error_stream)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # add all of the other operators (mostly irrelevant)
    # just use the front camera for all other components
    center_camera_stream = rgb_camera_streams[0]
    center_camera_setup = rgb_camera_setups[0]
    obstacles_stream = obstacles_streams[0]
    ground_segmented_stream = segmented_camera_streams[0]
    point_cloud_stream = point_cloud_streams[0]
    depth_stream = depth_streams[0]

    tl_transform = pylot.utils.Transform(FRONT_CAMERA_LOCATION,
                                        pylot.utils.Rotation())
    # = ground, = None
    traffic_lights_stream, tl_camera_stream = \
        pylot.component_creator.add_traffic_light_detection(
            tl_transform, vehicle_id_stream, release_sensor_stream,
            pose_stream, depth_stream, ground_traffic_lights_stream)

    # = None
    lane_detection_stream = pylot.component_creator.add_lane_detection(
        center_camera_stream, pose_stream, open_drive_stream)
    if lane_detection_stream is None:
        lane_detection_stream = erdos.IngestStream()
        streams_to_send_top_on.append(lane_detection_stream)

    # = None
    obstacles_tracking_stream = pylot.component_creator.add_obstacle_tracking(
        center_camera_stream, center_camera_setup, obstacles_stream,
        depth_stream, vehicle_id_stream, pose_stream, ground_obstacles_stream,
        time_to_decision_loop_stream)

    # = None
    segmented_stream = pylot.component_creator.add_segmentation(
        center_camera_stream, ground_segmented_stream)

    # = None
    depth_stream = pylot.component_creator.add_depth(front_transform,
                                                     vehicle_id_stream,
                                                     center_camera_setup,
                                                     depth_camera_stream)

    #IGNORE
    if FLAGS.fusion:
        pylot.operator_creator.add_fusion(pose_stream, obstacles_stream,
                                          depth_stream,
                                          ground_obstacles_stream)
    # = None, = None, = None
    prediction_stream, prediction_camera_stream, notify_prediction_stream = \
        pylot.component_creator.add_prediction(
            obstacles_tracking_stream, vehicle_id_stream, front_transform,
            release_sensor_stream, pose_stream, point_cloud_stream,
            lidar_setup)
    if prediction_stream is None:
        prediction_stream = obstacles_stream
    if notify_prediction_stream:
        notify_streams.append(notify_prediction_stream)

    goal_location = pylot.utils.Location(float(FLAGS.goal_location[0]),
                                         float(FLAGS.goal_location[1]),
                                         float(FLAGS.goal_location[2]))

    waypoints_stream = pylot.component_creator.add_planning(
        goal_location, pose_stream, prediction_stream, traffic_lights_stream,
        lane_detection_stream, open_drive_stream, global_trajectory_stream,
        time_to_decision_loop_stream)

    if FLAGS.simulator_mode == "pseudo-asynchronous":
        # Add a synchronizer in the pseudo-asynchronous mode.
        (
            waypoints_stream_for_control,
            pose_stream_for_control,
            sensor_ready_stream,
            _pipeline_finish_notify_stream,
        ) = pylot.operator_creator.add_planning_pose_synchronizer(
            waypoints_stream, pose_stream_for_control, pose_stream,
            *notify_streams)
        release_sensor_stream.set(sensor_ready_stream)
        pipeline_finish_notify_stream.set(_pipeline_finish_notify_stream)
    else:
        waypoints_stream_for_control = waypoints_stream
        pose_stream_for_control = pose_stream


    # synchronizing on front perfect obstacle stream -- is that correct?
    control_stream = pylot.component_creator.add_control(
        pose_stream_for_control, waypoints_stream_for_control,
        vehicle_id_stream, perfect_obstacles_streams[0])
    control_loop_stream.set(control_stream)

    add_evaluation_operators(vehicle_id_stream, pose_stream, imu_stream,
                             pose_stream_for_control,
                             waypoints_stream_for_control)

    time_to_decision_stream = pylot.operator_creator.add_time_to_decision(
        pose_stream, obstacles_streams[0])
    time_to_decision_loop_stream.set(time_to_decision_stream)

    control_display_stream = None
    if pylot.flags.must_visualize():
        control_display_stream, ingest_streams = \
            pylot.operator_creator.add_visualizer(
                pose_stream, rgb_camera_streams, bird_camera_stream, tl_camera_stream,
                prediction_camera_stream, depth_camera_streams,
                point_cloud_streams, segmented_stream, imu_stream,
                obstacles_streams, obstacles_error_streams, perfect_obstacles_streams,
                traffic_lights_stream, obstacles_tracking_stream, lane_detection_stream,
                prediction_stream, waypoints_stream, control_stream)
        streams_to_send_top_on += ingest_streams

    node_handle = erdos.run_async()

    for stream in streams_to_send_top_on:
        stream.send(erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))
    # If we did not use the pseudo-asynchronous mode, ask the sensors to
    # release their readings whenever.
    if FLAGS.simulator_mode != "pseudo-asynchronous":
        release_sensor_stream.send(
            erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))
        pipeline_finish_notify_stream.send(
            erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    return node_handle, control_display_stream


def shutdown_pylot(node_handle, client, world):
    if node_handle:
        node_handle.shutdown()
    else:
        print('The Pylot dataflow failed to initialize.')
    if FLAGS.simulation_recording_file is not None:
        client.stop_recorder()
    set_asynchronous_mode(world)
    if pylot.flags.must_visualize():
        import pygame
        pygame.quit()


def shutdown(sig, frame):
    raise KeyboardInterrupt


def main(args):
    # Connect an instance to the simulator to make sure that we can turn the
    # synchronous mode off after the script finishes running.
    client, world = get_world(FLAGS.simulator_host, FLAGS.simulator_port,
                              FLAGS.simulator_timeout)
    try:
        if FLAGS.simulation_recording_file is not None:
            client.start_recorder(FLAGS.simulation_recording_file)
        node_handle, control_display_stream = driver()
        signal.signal(signal.SIGINT, shutdown)
        if pylot.flags.must_visualize():
            pylot.utils.run_visualizer_control_loop(control_display_stream)
        node_handle.wait()
    except KeyboardInterrupt:
        shutdown_pylot(node_handle, client, world)
    except Exception:
        shutdown_pylot(node_handle, client, world)
        raise

if __name__ == '__main__':
    app.run(main)
