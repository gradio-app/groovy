<p align="center">
    <a href="https://github.com/abidlabs/grumpy/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/abidlabs/grumpy.svg?color=blue"></a>
    <a href="https://pypi.org/project/grumpy/"><img alt="PyPI" src="https://img.shields.io/pypi/v/grumpy"></a>
    <img alt="Python version" src="https://img.shields.io/badge/python-3.10+-important">
    <a href="https://github.com/abidlabs/grumpy/actions/workflows/format.yml"><img alt="Format" src="https://github.com/abidlabs/grumpy/actions/workflows/format.yml/badge.svg"></a>
    <a href="https://github.com/abidlabs/grumpy/actions/workflows/test.yml"><img alt="Test" src="https://github.com/abidlabs/grumpy/actions/workflows/test.yml/badge.svg"></a>
</p>


<h1 align="center">🕺 grumpy</h1>

Hello! This is `grumpy`, a Python library that makes it easy to build, debug, and share workflows (_flows_), e.g. autonomous applications that perform actions using your browser or desktop.

✨ **Build** flows with a simple high-level `Flow` class that can wrap any kind of application. Since `grumpy` comes "batteries-included", you can write your first Flow to control a browser in just a single line.

🔎 **Debug** flows with an intuitive Gradio user interface, that exposes agent thought while it runs and allows users to "step-in" and intervene at any point.

🤗 **Share** flows on Hugging Face Spaces publicly (or with specific collaborators) and reuse flows from the community


![Screen Recording 2025-01-29 at 1 30 30 AM (online-video-cutter com)](https://github.com/user-attachments/assets/6cb171cd-9a8a-41e2-927c-badf694595d4)

 
## Installation

Assuming you have Python 3.10 or higher already installed, run in your terminal:

```bash
pip install "grumpy[full]"
```

## Key Features

### 1. Get started immediately ✨

Run browser automations with a single terminal command. No need for complex setup or boilerplate code:

```bash
grumpy flow "Use Wikipedia to tell me the birth date of George Washington. Return the final answer in this format: MM-DD-YYYY."
```

Note: by default, grumpy uses a web-browsing agent built using the excellent [`smolagents` library](https://github.com/huggingface/smolagents). The web-browsing agent works best when provided detailed, step-by-step instructions.

### 2. Customize Flows

Create interactive automation apps using the `Flow` class. You can define input parameters that users can customize before running the flow:

```python
from grumpy import Flow
import gradio as gr

flow = Flow(
    task="Find the next upcoming meetup in {} related to {}",
    inputs=[
        gr.Textbox(label="Location", value="San Francisco")
        gr.Textbox(label="Activity", value="board games"),
    ]
)

flow.launch()
```

#### Run Flows Programmatically

The `Flow` class can also be run programmatically so that it can be used as part of larger programs. Here's an example:

```python
from grumpy import Flow
import csv

flow = Flow(task="Find the next upcoming meetup in {} related to {}")

cities = [
    "San Francisco", "New York", "Chicago", "Los Angeles", "Seattle",
    "Austin", "Boston", "Denver", "Portland", "Miami"
]

results = []
for city in cities:
    event_info = flow.run(city, "board games")
    results.append({"city": city, "event": event_info})

with open("board_game_events.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["city", "event"])
    writer.writeheader()
    writer.writerows(results)
```

### 3. Easy Sharing via Hugging Face Spaces

Share your automation workflows with others by publishing to Hugging Face Spaces. Just navigate to your project folder and run in the terminal:

```bash
grumpy publish
```

This will create a public (you can change visibility to private) Hugging Face Space where others can access and use your automation.

### 4. Use Community Workflows

Take advantage of existing workflows created by the community. Run any published workflow locally from your terminal, e.g.:

```bash
grumpy flow https://huggingface.co/spaces/abidlabs/Activity_Finder
```

The Python equivalent to load a Flow from Spaces is also straightforward:

```python
import grumpy as gv

flow = gv.Flow.from_space("https://huggingface.co/spaces/abidlabs/Activity_Finder")

...
```

## Roadmap aka leftover TODOs

* Get a complex flow woking reliably and fast, add logs
* Improve default BrowserAgent tool as it gives lookup errors way too often.
* Generally improve troubleshooting
* Better cleanup of gradio ui and browsing windows when script is terminated
* Allow pausing / stopping from gradio ui / include timing information in UI
* Warn before using OPENAI_API_KEY / HF_TOKEN
* Add better examples in readme / Spaces (twitter explorer find tweets about a particular topic), research presidents' spouse. Chain Flows together
* Add support for using the user's default browser 

* Allow retrying from cli and python (understand why browsing fails first. also add max_time_per_step?)
* Support `max_steps` / `max_time` in `.run()`
* Be smarter about screenshots
* Autogenerated docs

* Support structured generation in `.run()` -- how would this work?
* Make it easier to modify the default agent (e.g. by copy-pasting it into the working directory)
* Allow `task` to be an arbitrary function of inputs, not just a format string
* Add support for `browser-use` and desktop apps
* Give examples of how this could be useful long-term (e.g. autonomous agents running on cron jobs & slacking you important stuff)
* Figure out a way to get human input

## Contributing

Contributions are welcome! Feel free to submit bug reports and feature requests or submit pull requests. Star this repo if you find it interesting ⭐

