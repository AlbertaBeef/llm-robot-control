import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import csv
import os
import time

from ament_index_python.packages import get_package_share_directory, get_package_prefix
from pathlib import Path

from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from rclpy.wait_for_message import wait_for_message

# Import the generated service type
try:
    from ros2_ai_interfaces.srv import SetLlmMode
except Exception:
    SetLlmMode = None
# Contents of srv/SetLlmMode.srv:
#    bool enable
#    string llm_api
#    string llm_model
#    ---
#    bool success
#    string message

class Ros2AIEval(Node):
    def __init__(self):
        super().__init__('ros2_ai_eval')
        self.get_logger().info('ROS2 AI Agent Evaluator has been started')
        
        # Parameters
        #
        self.declare_parameter("models_csv", "models.csv")
        self.models_csv = self.get_parameter("models_csv").value
        self.get_logger().info('models_csv : "%s"' % self.models_csv)
        #
        self.declare_parameter("tasks_csv", "tasks.csv")
        self.tasks_csv = self.get_parameter("tasks_csv").value
        self.get_logger().info('tasks_csv : "%s"' % self.tasks_csv)
        #
        self.declare_parameter("results_csv", "results.csv")
        self.results_csv = self.get_parameter("results_csv").value
        self.get_logger().info('results_csv : "%s"' % self.results_csv)
        #
        self.declare_parameter("task_delay", 1.0)
        self.task_delay = self.get_parameter("task_delay").value
        self.get_logger().info('task_delay : "%f"' % self.task_delay)

        # Publishers and Subscribers
        self.pub_llm_prompt = self.create_publisher(String, '/llm_prompt', 10)
        self.sub_llm_tool_calls = self.create_subscription(String, '/llm_tool_calls', self.llm_tool_calls_callback, 10)
        self.sub_llm_output = self.create_subscription(String, '/llm_output', self.llm_output_callback, 10)

        # LLM configuration service
        if SetLlmMode is not None:
            self.set_llm_cli = self.create_client(
                SetLlmMode,
                'set_llm_mode'
            )
            #while not self.set_llm_cli.wait_for_service(timeout_sec=1.0):
            #    self.get_logger().info('Waiting for llm_set_mode service...')
            
            self.get_logger().info('LLM state service /set_llm_mode ready (type: ros2_ai_interfaces/srv/SetLlmMode)')
        else:
            self.get_logger().warn(
                'Service /SetLlmMode not available. Build and install the ros2_ai_interfaces packageto enable the service.'
            )

        
        # Load CSV files
        self.models = self.read_models()
        self.nb_models = len(self.models)
        self.get_logger().info("models = ")
        for i,model in enumerate(self.models):
            self.get_logger().info(f"    {model}")
        
        self.tasks = self.read_tasks()
        self.nb_tasks = len(self.tasks)
        self.get_logger().info("[INFO] tasks = ")
        for i,task in enumerate(self.tasks):
            self.get_logger().info(f"    {task}")

        # Result buffer
        self.results = []
        self.llm_tool_calls = []
        self.llm_output = []

        # Timer for one-shot delays
        self.task_delay_timer = None

        # Start evaluation
        self.model_id = 0
        self.task_id = 0
        self.evaluate_task()

    def read_models(self):
        models = []
        with open(self.models_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                models.append({'llm_api': row['llm_api'], 'llm_model': row['llm_model']})
        return models

    def read_tasks(self):
        tasks = []
        with open(self.tasks_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tasks.append({'llm_prompt': row['llm_prompt']})
        return tasks

    def evaluate_task(self):
        self.model_dict = self.models[self.model_id]
        self.task_dict = self.tasks[self.task_id]

        self.get_logger().info(f"Evaluating model[{self.model_id}]={self.model_dict}, with task[{self.task_id}]={self.task_dict} ...")
        
        if self.task_id == 0:
            self.llm_api = self.model_dict['llm_api']
            self.llm_model = self.model_dict['llm_model']
            self.get_logger().info(f"Configuring agent for model[{self.model_id}]={self.model_dict} ...")
            # Use async call instead
            self.set_llm_mode_async(self.llm_api, self.llm_model)
            return  # Early return, will continue in callback

        """Continue with the current task (no need to configure LLM)"""
        self.llm_tool_calls = []
        self.llm_prompt = self.task_dict['llm_prompt']
        self.publish_prompt(self.llm_prompt)

    def set_llm_mode_async(self, api, model):
        """Async version that doesn't block"""
        if SetLlmMode is not None:
            req = SetLlmMode.Request()
            req.enable = True
            req.llm_api = api
            req.llm_model = model
            future = self.set_llm_cli.call_async(req)
            # Add callback to continue when done
            future.add_done_callback(lambda f: self.on_llm_configured(f, api, model))
        else:
            self.get_logger().warn('Service /SetLlmMode not available.')
            # Continue anyway
            self.continue_evaluation()

    def on_llm_configured(self, future, api, model):
        """Called when LLM configuration service completes"""
        try:
            result = future.result()
            if hasattr(result, 'success') and result.success:
                self.get_logger().info(f"set_llm_mode: llm_enable={True} | llm_api={api} | llm_model={model}")
            else:
                self.get_logger().error(f"Failed to set_llm_mode: llm_enable={True} | llm_api={api} | llm_model={model}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

        # Continue immediately - service already ensured agent is configured
        self.continue_evaluation()

    def continue_evaluation(self):
        """Continue with the current task after LLM is configured"""
        self.llm_tool_calls = []
        self.llm_prompt = self.task_dict['llm_prompt']
        self.publish_prompt(self.llm_prompt)

    def publish_prompt(self, prompt):
        msg = String()
        msg.data = prompt
        self.pub_llm_prompt.publish(msg)
        self.get_logger().info(f"Published prompt: {prompt}")

    def llm_tool_calls_callback(self, msg):
        self.get_logger().info(f"Received message from llm_tool_calls: {msg.data}")
        self.llm_tool_calls.append(msg.data)

    def llm_output_callback(self, msg):
        self.get_logger().info(f"Received message from llm_output: {msg.data}")
        self.llm_output = msg.data
        
        #fieldnames = ['llm_api', 'llm_model', 'llm_prompt', 'llm_tool_calls', 'llm_output']
        result = {
            'llm_api':self.llm_api,
            'llm_model':self.llm_model,
            'llm_prompt':self.llm_prompt,
            'llm_tool_calls':self.llm_tool_calls,
            'llm_output':self.llm_output
        }
        self.get_logger().info(f"Results : {result}")
        self.results.append(result)
                
        self.task_id = self.task_id + 1
        if self.task_id == self.nb_tasks:
            # Last task, move to next model
            self.task_id = 0
            self.model_id = self.model_id + 1
            if self.model_id == self.nb_models:
                # Last model, evaluation complete
                self.get_logger().info('Writing results ...')
                self.write_results()
                self.get_logger().info('=======================================')
                self.get_logger().info('=====   Evaluation complete !     =====')
                self.get_logger().info('=======================================')
                #rclpy.shutdown()
                return

        if self.task_delay > 0.0:
            # Add task delay before continuing to next task
            self.get_logger().info(f"Task Delay : Waiting {self.task_delay} seconds before next task...")
            self.task_delay_timer = self.create_timer(self.task_delay, self.proceed_to_next_task)
        else:
            # Proceed to next task without delay
            self.evaluate_task()

    def proceed_to_next_task(self):
        """Called after task delay to continue evaluation"""
        self.get_logger().info(f"Task Delay : done")        
        # Cancel and destroy the timer (one-shot behavior)
        if self.task_delay_timer is not None:
            self.task_delay_timer.cancel()
            self.destroy_timer(self.task_delay_timer)
            self.task_delay_timer = None

        # Proceed to next task after delay
        self.evaluate_task()

    def write_results(self):
        fieldnames = ['llm_api', 'llm_model', 'llm_prompt', 'llm_tool_calls', 'llm_output']
        with open(self.results_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result)
        self.get_logger().info(f"Wrote results to {self.results_csv}")

def main(args=None):
    rclpy.init(args=args)
    node = Ros2AIEval()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
