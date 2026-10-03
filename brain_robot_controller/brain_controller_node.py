import os
import joblib
import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Int32, Float32
from geometry_msgs.msg import Twist


class BrainControllerNode(Node):

    def __init__(self):
        super().__init__('brain_controller_node')

        # =================================================
        # PROJECT PATHS
        # =================================================

        # Source package directory:
        # ~/brain_robot_ws/src/brain_robot_controller

        package_dir = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        # ANN model
        model_path = os.path.join(
            package_dir,
            'models',
            'ann_model.joblib'
        )

        # Real HAPT test dataset
        data_path = os.path.join(
            package_dir,
            'data',
            'processed',
            'X_test.npy'
        )

        self.get_logger().info(
            'Loading ANN model...'
        )

        # Load trained ANN
        self.model = joblib.load(
            model_path
        )

        # Load real-world HAPT test data
        self.X_test = np.load(
            data_path
        )

        self.get_logger().info(
            f'ANN model loaded. '
            f'Test samples: {len(self.X_test)}'
        )

        # =================================================
        # HAPT ACTIVITY LABELS
        # =================================================

        self.activities = [
            'WALKING',
            'WALKING_UPSTAIRS',
            'WALKING_DOWNSTAIRS',
            'SITTING',
            'STANDING',
            'LAYING',
            'STAND_TO_SIT',
            'SIT_TO_STAND',
            'SIT_TO_LIE',
            'LIE_TO_SIT',
            'STAND_TO_LIE',
            'LIE_TO_STAND'
        ]

        # =================================================
        # TiO2 MEMRISTOR PARAMETERS
        # =================================================

        self.G_on = 1e-3
        self.G_off = 1e-5

        self.V_threshold = 0.20

        # Improved dynamics
        self.k_positive = 1.2
        self.k_negative = 1.2

        # Initial internal state
        self.memristor_state = 0.05

        # ROS control interval
        self.dt = 0.10
        self.memristor_dt = 0.001

        # =================================================
        # SPIKE PARAMETERS
        # =================================================

        self.max_rate = 100.0

        # Reproducible random generator
        self.rng = np.random.default_rng(
            42
        )

        # =================================================
        # DATASET CONTROL
        # =================================================

        self.sample_index = 0

        # =================================================
        # ROS 2 PUBLISHERS
        # =================================================

        self.activity_pub = self.create_publisher(
            String,
            '/brain/activity',
            10
        )

        self.activity_id_pub = self.create_publisher(
            Int32,
            '/brain/activity_id',
            10
        )

        self.spike_rate_pub = self.create_publisher(
            Float32,
            '/brain/spike_rate',
            10
        )

        self.memristor_state_pub = self.create_publisher(
            Float32,
            '/brain/memristor_state',
            10
        )

        self.memristor_current_pub = self.create_publisher(
            Float32,
            '/brain/memristor_current',
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # =================================================
        # REAL-TIME CONTROL TIMER
        # =================================================

        self.timer = self.create_timer(
            self.dt,
            self.control_loop
        )

        self.get_logger().info(
            'Brain-inspired controller started.'
        )

        self.get_logger().info(
            'Pipeline: '
            'Real HAPT Dataset -> ANN -> '
            'Spike Generator -> TiO2 Memristor -> Robot'
        )

    # =====================================================
    # TiO2 MEMRISTOR MODEL
    # =====================================================

    def update_memristor(
        self,
        voltage
    ):

        w = self.memristor_state

        # Positive voltage above threshold
        if voltage > self.V_threshold:

            dw_dt = (
                self.k_positive
                * (
                    voltage
                    - self.V_threshold
                )
                * (
                    1.0 - w
                )
            )

        # Negative voltage below threshold
        elif voltage < -self.V_threshold:

            dw_dt = (
                self.k_negative
                * (
                    voltage
                    + self.V_threshold
                )
                * w
            )

        # No state change below threshold
        else:

            dw_dt = 0.0

        # Euler integration
        w = w + (
            dw_dt
            * self.memristor_dt
        )

        # Keep state between 0 and 1
        w = np.clip(
            w,
            0.0,
            1.0
        )

        self.memristor_state = w

        # Conductance
        conductance = (
            self.G_off
            + w
            * (
                self.G_on
                - self.G_off
            )
        )

        # Memristor current
        current = (
            conductance
            * voltage
        )

        return (
            conductance,
            current
        )

    # =====================================================
    # ROBOT CONTROL
    # =====================================================

    def generate_robot_command(
        self,
        activity
    ):

        cmd = Twist()

        # =================================================
        # ACTIVITY-BASED ROBOT MOVEMENT
        # =================================================

        # Normal walking -> move forward
        if activity == 'WALKING':

            cmd.linear.x = 0.30
            cmd.angular.z = 0.0

        # Walking upstairs -> faster forward + left turn
        elif activity == 'WALKING_UPSTAIRS':

            cmd.linear.x = 0.35
            cmd.angular.z = 0.25

        # Walking downstairs -> slower forward + right turn
        elif activity == 'WALKING_DOWNSTAIRS':

            cmd.linear.x = 0.15
            cmd.angular.z = -0.25

        # Standing -> rotate slowly and search
        elif activity == 'STANDING':

            cmd.linear.x = 0.0
            cmd.angular.z = 0.30

        # Sitting -> stop
        elif activity == 'SITTING':

            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        # Laying -> stop
        elif activity == 'LAYING':

            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        # Sitting -> standing -> start moving slowly
        elif activity == 'SIT_TO_STAND':

            cmd.linear.x = 0.10
            cmd.angular.z = 0.0

        # Lying -> standing -> start moving slowly
        elif activity == 'LIE_TO_STAND':

            cmd.linear.x = 0.10
            cmd.angular.z = 0.0

        # Lying -> sitting -> start moving slowly
        elif activity == 'LIE_TO_SIT':

            cmd.linear.x = 0.08
            cmd.angular.z = 0.0

        # Standing -> sitting -> stop
        elif activity == 'STAND_TO_SIT':

            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        # Sitting -> lying -> stop
        elif activity == 'SIT_TO_LIE':

            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        # Standing -> lying -> stop
        elif activity == 'STAND_TO_LIE':

            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        # Any unexpected activity -> safe stop
        else:

            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        return cmd

    # =====================================================
    # MAIN REAL-TIME CONTROL LOOP
    # =====================================================

    def control_loop(self):

        # -------------------------------------------------
        # STEP 1: GET REAL HAPT DATASET SAMPLE
        # -------------------------------------------------

        sample = self.X_test[
            self.sample_index
        ]

        # -------------------------------------------------
        # STEP 2: ANN PREDICTION
        # -------------------------------------------------

        probabilities = (
            self.model.predict_proba(
                sample.reshape(
                    1,
                    -1
                )
            )[0]
        )

        predicted_class = int(
            np.argmax(
                probabilities
            )
        )

        confidence = float(
            probabilities[
                predicted_class
            ]
        )

        activity = self.activities[
            predicted_class
        ]

        # -------------------------------------------------
        # STEP 3: ANN CONFIDENCE -> SPIKE RATE
        # -------------------------------------------------

        firing_rate = (
            confidence
            * self.max_rate
        )

        # Probability of spike
        # during this timestep
        probability = (
            firing_rate
            * self.dt
        )

        probability = min(
            probability,
            1.0
        )

        # Generate stochastic spike
        spike = (
            self.rng.random()
            < probability
        )

        # -------------------------------------------------
        # STEP 4: SPIKE -> TiO2 VOLTAGE
        # -------------------------------------------------

        if spike:

            voltage = 0.50

        else:

            voltage = 0.0

        # -------------------------------------------------
        # STEP 5: UPDATE TiO2 MEMRISTOR
        # -------------------------------------------------

        (
            conductance,
            current
        ) = self.update_memristor(
            voltage
        )

        # -------------------------------------------------
        # STEP 6: GENERATE ROBOT COMMAND
        # -------------------------------------------------

        cmd = (
            self.generate_robot_command(
                activity
            )
        )

        self.cmd_vel_pub.publish(
            cmd
        )

        # -------------------------------------------------
        # STEP 7: PUBLISH ACTIVITY
        # -------------------------------------------------

        activity_msg = String()

        activity_msg.data = (
            activity
        )

        self.activity_pub.publish(
            activity_msg
        )

        # -------------------------------------------------
        # STEP 8: PUBLISH ACTIVITY ID
        # -------------------------------------------------

        id_msg = Int32()

        id_msg.data = (
            predicted_class + 1
        )

        self.activity_id_pub.publish(
            id_msg
        )

        # -------------------------------------------------
        # STEP 9: PUBLISH SPIKE RATE
        # -------------------------------------------------

        rate_msg = Float32()

        rate_msg.data = float(
            firing_rate
        )

        self.spike_rate_pub.publish(
            rate_msg
        )

        # -------------------------------------------------
        # STEP 10: PUBLISH MEMRISTOR STATE
        # -------------------------------------------------

        state_msg = Float32()

        state_msg.data = float(
            self.memristor_state
        )

        self.memristor_state_pub.publish(
            state_msg
        )

        # -------------------------------------------------
        # STEP 11: PUBLISH MEMRISTOR CURRENT
        # -------------------------------------------------

        current_msg = Float32()

        current_msg.data = float(
            current
        )

        self.memristor_current_pub.publish(
            current_msg
        )

        # -------------------------------------------------
        # STEP 12: REAL-TIME RESULT
        # -------------------------------------------------

        self.get_logger().info(
            f'Activity: '
            f'{activity:18s} | '
            f'Confidence: '
            f'{confidence * 100:6.2f}% | '
            f'Rate: '
            f'{firing_rate:6.2f} Hz | '
            f'Spike: '
            f'{int(spike)} | '
            f'TiO2 state: '
            f'{self.memristor_state:.4f} | '
            f'G: '
            f'{conductance:.6f} S | '
            f'I: '
            f'{current:.6f} A'
        )

        # -------------------------------------------------
        # STEP 13: NEXT DATASET SAMPLE
        # -------------------------------------------------

        self.sample_index += 1

        if (
            self.sample_index
            >= len(self.X_test)
        ):

            self.sample_index = 0

            self.get_logger().info(
                'Completed one full '
                'HAPT test-dataset cycle.'
            )


# =========================================================
# MAIN FUNCTION
# =========================================================

def main(
    args=None
):

    rclpy.init(
        args=args
    )

    node = BrainControllerNode()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        node.get_logger().info(
            'Controller stopped by user.'
        )

    finally:

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == '__main__':

    main()
