import sys
import os
import gradio as gr
import json

# Add the path to midigpt
sys.path.append(os.path.dirname(os.getcwd()) + "/python_lib")
import midigpt

def generate_midi(midi_input, ckpt, out, temperature, instrument, density, track_type, polyphony_hard_limit, shuffle, verbose, max_steps, batch_size, percentage, model_dim, mask_top_k, sampling_seed, autoregressive):
    # Save uploaded MIDI file to a temp path
    midi_input_path = midi_input.name
    if out:
        midi_dest = out
    else:
        midi_dest = os.path.join(os.path.split(midi_input_path)[0], 'midigpt_gen.mid')
    e = midigpt.ExpressiveEncoder()
    midi_json_input = json.loads(e.midi_to_json(midi_input_path))
    valid_status = {'tracks': [
        {
            'track_id': 0,
            'temperature': temperature,
            'instrument': instrument,
            'density': density,
            'track_type': track_type,
            'ignore': False,
            'selected_bars': [False, False, True, False],
            'min_polyphony_q': 'POLYPHONY_ANY',
            'max_polyphony_q': 'POLYPHONY_ANY',
            'autoregressive': autoregressive,
            'polyphony_hard_limit': polyphony_hard_limit
        }
    ]}
    parami = {
        'tracks_per_step': 1,
        'bars_per_step': 1,
        'model_dim': model_dim,
        'percentage': percentage,
        'batch_size': batch_size,
        'temperature': temperature,
        'max_steps': max_steps,
        'polyphony_hard_limit': polyphony_hard_limit,
        'shuffle': shuffle,
        'verbose': verbose,
        'ckpt': ckpt,
        'sampling_seed': sampling_seed,
        'mask_top_k': mask_top_k
    }
    piece = json.dumps(midi_json_input)
    status = json.dumps(valid_status)
    param = json.dumps(parami)
    callbacks = midigpt.CallbackManager()
    max_attempts = 3
    midi_str = midigpt.sample_multi_step(piece, status, param, max_attempts, callbacks)[0]
    e = midigpt.ExpressiveEncoder()
    e.json_to_midi(midi_str, midi_dest)
    return midi_dest

def main():
    iface = gr.Interface(
        fn=generate_midi,
        inputs=[
            gr.File(label="Input MIDI File"),
            gr.Textbox(label="Checkpoint Path", value="../models/EXPRESSIVE_ENCODER_RES_1920_12_GIGAMIDI_CKPT_150K.pt"),
            gr.Textbox(label="Output MIDI Path (optional)", value=""),
            gr.Slider(0.0, 2.0, value=0.5, label="Temperature"),
            gr.Textbox(label="Instrument", value="acoustic_grand_piano"),
            gr.Slider(1, 20, value=10, step=1, label="Density"),
            gr.Slider(0, 20, value=10, step=1, label="Track Type"),
            gr.Slider(1, 16, value=9, step=1, label="Polyphony Hard Limit"),
            gr.Checkbox(label="Shuffle", value=True),
            gr.Checkbox(label="Verbose", value=True),
            gr.Slider(1, 1000, value=200, step=1, label="Max Steps"),
            gr.Slider(1, 32, value=1, step=1, label="Batch Size"),
            gr.Slider(1, 100, value=100, step=1, label="Percentage"),
            gr.Slider(1, 32, value=4, step=1, label="Model Dim"),
            gr.Slider(0, 100, value=0, step=1, label="Mask Top K"),
            gr.Number(label="Sampling Seed", value=-1),
            gr.Checkbox(label="Autoregressive", value=False)
        ],
        outputs=gr.File(label="Generated MIDI File"),
        title="MIDI-GPT Generator",
        description="Generate expressive MIDI using MIDI-GPT."
    )
    iface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
