import gradio as gr

from grumpy.flow import Flow


def test_flow_with_simple_task():
    """Test Flow with a basic task string"""
    flow = Flow(task="Hello World", agent_fn=lambda x: x)
    assert flow.task == "Hello World"
    assert len(flow.inputs) == 0
    assert flow.agent_fn is not None


def test_flow_with_gradio_inputs():
    """Test Flow initialization with Gradio input components"""
    text_input = gr.Textbox(label="Input")
    slider = gr.Slider(minimum=0, maximum=100)
    flow = Flow(
        task="Process {text} with value {value}",
        inputs=[text_input, slider],
        agent_fn=lambda x: x,
    )

    assert len(flow.inputs) == 2
    assert isinstance(flow.inputs[0], gr.Textbox)
    assert isinstance(flow.inputs[1], gr.Slider)


def test_flow_serialization():
    """Test Flow serialization to and from JSON"""
    text_input = gr.Textbox(label="Test Input")
    original_flow = Flow(
        task="Process {text}", inputs=[text_input], agent_fn=lambda x: x
    )

    json_data = original_flow.to_json()
    restored_flow = Flow.from_json(json_data, agent_fn=lambda x: x)

    assert restored_flow.task == original_flow.task
    assert len(restored_flow.inputs) == len(original_flow.inputs)
    assert isinstance(restored_flow.inputs[0], gr.Textbox)
    assert restored_flow.inputs[0].label == "Test Input"
