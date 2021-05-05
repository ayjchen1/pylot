"""Implements an operator that eveluates detection output."""
import heapq
import time

import erdos
from erdos import Message, ReadStream, Timestamp, WriteStream

from pylot.perception.messages import ObstaclesMessage
import pylot.perception.detection.utils
from pylot.utils import time_epoch_ms


class DetectionEvalOperator(erdos.Operator):
    """Operator that computes accuracy metrics using detected obstacles.

    Args:
        obstacles_stream (:py:class:`erdos.ReadStream`): The stream on which
            detected obstacles are received.
        ground_obstacles_stream: The stream on which
            :py:class:`~pylot.perception.messages.ObstaclesMessage` are
            received from the simulator.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, obstacles_stream, ground_obstacles_stream, obstacles_error_stream, flags):
        obstacles_stream.add_callback(self.on_obstacles)
        ground_obstacles_stream.add_callback(self.on_ground_obstacles)
        erdos.add_watermark_callback(
            [obstacles_stream, ground_obstacles_stream], [obstacles_error_stream], self.on_watermark)
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        self._last_notification = None
        # Buffer of detected obstacles.
        self._detected_obstacles = []
        # Buffer of ground obstacles.
        self._ground_obstacles = []
        # Heap storing pairs of (ground/output time, game time).
        self._detector_start_end_times = []
        self._sim_interval = None

    @staticmethod
    def connect(obstacles_stream, ground_obstacles_stream):
        """Connects the operator to other streams.

        Args:
            obstacles_stream (:py:class:`erdos.ReadStream`): The stream
                on which detected obstacles are received.
            ground_obstacles_stream: The stream on which
                :py:class:`~pylot.perception.messages.ObstaclesMessage` are
                received from the simulator.
        """
        obstacles_error_stream = erdos.WriteStream()
        return [obstacles_error_stream]

    def on_watermark(self, timestamp, obstacles_error_stream):
        """Invoked when all input streams have received a watermark.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        assert len(timestamp.coordinates) == 1
        op_start_time = time.time()
        game_time = timestamp.coordinates[0]
        if not self._last_notification:
            self._last_notification = game_time
            return
        else:
            self._sim_interval = (game_time - self._last_notification)
            self._last_notification = game_time

        sim_time = timestamp.coordinates[0]
        while len(self._detector_start_end_times) > 0:
            print("INSIDE EVAL WATERMARK", timestamp)
            (end_time, start_time) = self._detector_start_end_times[0]

            if end_time <= game_time:
                heapq.heappop(self._detector_start_end_times)

                ego_transform, ground_obstacles = self.__get_ground_obstacles_at(end_time)

                #go_labels = []
                #for go in ground_obstacles:
                #    go_labels.append(go.label)

                obstacles = self.__get_obstacles_at(start_time)

                #do_labels = []
                #for do in obstacles:
                #    do_labels.append(do.label)
                #print(timestamp, go_labels, do_labels)

                if (len(obstacles) > 0 or len(ground_obstacles) > 0):
                    errs = pylot.perception.detection.utils.get_errors(
                        ground_obstacles, obstacles, ego_transform)

                    # Get runtime in ms
                    runtime = (time.time() - op_start_time) * 1000
                    # self._csv_logger.info('{},{},{},{},{:.4f}'.format(
                    #    time_epoch_ms(), sim_time, self.config.name, 'runtime',
                    #    runtime))

                    self._logger.info('errors calculated')

                    matchobs = []
                    for i in range(len(errs)):
                        ground_ob = errs[i][0]
                        det_ob = errs[i][1]
                        err_val = errs[i][2]

                        det_id = "NONE"
                        if (det_ob is not None):
                            det_ob.vis_error = err_val
                            det_id = det_ob.id

                        ego_point = ego_transform.location.as_numpy_array().reshape(1, 3)
                        ego_loc = ego_transform.inverse_transform_points(ego_point).reshape(3,)

                        ob_actual_point = ground_ob.transform.location.as_numpy_array().reshape(1, 3)
                        ob_actual_loc = ego_transform.inverse_transform_points(ob_actual_point).reshape(3,)

                        relative_dist = ob_actual_loc - ego_loc

                        #print([ground_ob.id, ego_loc, "===", ob_actual_loc, "====", relative_dist])
                        print([ground_ob.id, det_id, err_val])
                        
                        self._csv_logger.info('{},{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f}'.format(
                            time_epoch_ms(), sim_time, self.config.name, ground_ob.id, ground_ob.label,
                            relative_dist[0], relative_dist[1], relative_dist[2],
                            err_val))

                self._logger.debug('Computing accuracy for {} {}'.format(
                    end_time, start_time))

                obstacles_error_stream.send(ObstaclesMessage(timestamp, obstacles))
                obstacles_error_stream.send(erdos.WatermarkMessage(timestamp))
            else:
                # The remaining entries require newer ground obstacles.
                break

        self.__garbage_collect_obstacles()

    def __get_ground_obstacles_at(self, timestamp):
        for (ground_time, obstacles, ego_transform) in self._ground_obstacles:
            if ground_time == timestamp:
                return ego_transform, obstacles
            elif ground_time > timestamp:
                break
        self._logger.fatal(
            'Could not find ground obstacles for {}'.format(timestamp))

    def __get_obstacles_at(self, timestamp):
        for (ground_time, obstacles) in self._detected_obstacles:
            if ground_time == timestamp:
                return obstacles
            elif ground_time > timestamp:
                break
        self._logger.fatal(
            'Could not find detected obstacles for {}'.format(timestamp))

    def __garbage_collect_obstacles(self):
        # Get the minimum watermark.
        watermark = None
        for (_, start_time) in self._detector_start_end_times:
            if watermark is None or start_time < watermark:
                watermark = start_time
        if watermark is None:
            return
        # Remove all detected obstacles that are below the watermark.
        index = 0
        while (index < len(self._detected_obstacles)
               and self._detected_obstacles[index][0] < watermark):
            index += 1
        if index > 0:
            self._detected_obstacles = self._detected_obstacles[index:]
        # Remove all the ground obstacles that are below the watermark.
        index = 0
        while (index < len(self._ground_obstacles)
               and self._ground_obstacles[index][0] < watermark):
            index += 1
        if index > 0:
            self._ground_obstacles = self._ground_obstacles[index:]

    def on_obstacles(self, msg):
        self._logger.debug('@{}: {} received obstacles'.format(
            msg.timestamp, self.config.name))
        game_time = msg.timestamp.coordinates[0]
        vehicles, people, _ = self.__get_obstacles_by_category(msg.obstacles)
        self._detected_obstacles.append((game_time, vehicles + people))
        # Two metrics: 1) mAP, and 2) timely-mAP
        if self._flags.detection_metric == 'mAP':
            # We will compare the obstacles with the ground truth at the same
            # game time.
            heapq.heappush(self._detector_start_end_times,
                           (game_time, game_time))
        elif self._flags.detection_metric == 'timely-mAP':
            # Ground obstacles time should be as close as possible to the time
            # of the obstacles + detector runtime.
            ground_obstacles_time = self.__compute_closest_frame_time(
                game_time + msg.runtime)
            # Round time to nearest frame.
            heapq.heappush(self._detector_start_end_times,
                           (ground_obstacles_time, game_time))
        else:
            raise ValueError('Unexpected detection metric {}'.format(
                self._flags.detection_metric))

    def on_ground_obstacles(self, msg):
        self._logger.debug('@{}: {} received ground obstacles'.format(
            msg.timestamp, self.config.name))
        game_time = msg.timestamp.coordinates[0]
        ego_transform = msg.ego_transform
        vehicles, people, _ = self.__get_obstacles_by_category(msg.obstacles)
        self._ground_obstacles.append((game_time, people + vehicles, ego_transform))

    def __compute_closest_frame_time(self, time):
        base = int(time) / self._sim_interval * self._sim_interval
        if time - base < self._sim_interval / 2:
            return base
        else:
            return base + self._sim_interval

    def __get_obstacles_by_category(self, obstacles):
        """Divides perception.detection.obstacle.Obstacle by labels."""
        vehicles = []
        people = []
        traffic_lights = []
        for obstacle in obstacles:
            if obstacle.is_vehicle():
                vehicles.append(obstacle)
            elif obstacle.is_person():
                people.append(obstacle)
            elif obstacle.is_traffic_light():
                traffic_lights.append(obstacle)
            else:
                self._logger.warning('Unexpected label {}'.format(
                    obstacle.label))
        return vehicles, people, traffic_lights
