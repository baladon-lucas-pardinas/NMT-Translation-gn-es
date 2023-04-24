import argparse
import os
import subprocess
import marian
import dotenv
import constants

def load_env() -> dict:
    # (Assumes env variables are paths)
    root_abs_dir = os.path.join(os.path.dirname(__file__), '..')
    env_abs_dir = os.path.join(root_abs_dir, '.env')
    dotenv.load_dotenv(env_abs_dir)

    # Loads env variables
    variable_names = constants.ENVIRONMENT_VARIABLES
    environment_variables = {
        variable_name.lower(): os.getenv(variable_name)
            for variable_name in variable_names
    }

    # Adds ../ to paths
    environment_variables = {
        variable_name: os.path.join(root_abs_dir, variable_value)
            for variable_name, variable_value in environment_variables.items()
                if variable_value is not None
    }
    return environment_variables

def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Train a marianMT model")
    return vars(parser.parse_args())

def execute_command(command: str) -> None:
    os.system(command)
    output = subprocess.check_output(command, shell=True)
    print(output.decode())

def main() -> None:
    environment_variables = load_env()
    args = parse_args()
    print(environment_variables, args)

    marian.train(**environment_variables, **args)

if __name__ == '__main__':
    main()