import subprocess
import webbrowser
import os
import time


def run_python_script_type3(script_path):
    try:
        # Start the subprocess
        process = subprocess.Popen(['python', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Wait for the process to finish and get the exit code
        rc = process.poll()

        # Print any remaining stderr messages
        stderr_output, _ = process.communicate()
        if stderr_output:
            print(stderr_output.strip())

        return rc
    except Exception as e:
        print(f"An error occurred: {e}")



def run_python_script(script_path):
    try:
        result = subprocess.run(['python', script_path], check=True, capture_output=True, text=True)
        print(f"Script Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running script: {e.stderr}")

def open_html_file(html_file_path):
    if os.path.exists(html_file_path):
        webbrowser.open(f'file://{os.path.abspath(html_file_path)}')
    else:
        print(f"HTML file does not exist: {html_file_path}")


python_script_path = './server/server.py' 
html_file_path = './client/index.html'    

open_html_file(html_file_path)
run_python_script_type3(python_script_path)



# The difference between "run_python_script" and "run_python_script_type3" functions is that the later prints some of the outputs of the server.py and former doesn't. 
# In terms of functionality, they do the same task.


# Needed for Pycharm... 
print("Press Ctrl+C to exit.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nExiting...")
