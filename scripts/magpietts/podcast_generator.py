import argparse
from nemo.collections.tts.models import MagpieTTSModel
from nemo.collections.tts.data.text_to_speech_dataset import MagpieTTSDataset, DatasetSample
from omegaconf.omegaconf import OmegaConf, open_dict
import torch
import os
import soundfile as sf
import numpy as np
import os
import json
from nemo_text_processing.text_normalization.normalize import Normalizer

def update_config(model_cfg, codecmodel_path, legacy_text_conditioning=False):
    model_cfg.codecmodel_path = codecmodel_path
    model_cfg.train_ds = None
    model_cfg.validation_ds = None
    model_cfg.legacy_text_conditioning = legacy_text_conditioning

    if "text_context_remapping_json" in model_cfg:
        del model_cfg.text_context_remapping_json

    return model_cfg

def get_audio_duration(file_path):
    with sf.SoundFile(file_path) as audio_file:
        # Calculate the duration
        duration = len(audio_file) / audio_file.samplerate
        return duration

def create_record(text, context_audio_filepath=None, context_text=None):
    text = text.strip()
    text = text_normalizer.normalize(text, punct_pre_process=True, punct_post_process=True)
    if len(text.split()) == 1:
        # Only one word, add a space in the beginning to avoid issues with single-word inputs
        text = " " + text

    # If transcript does not end with a period, question mark etc., add a period
    if text and text[-1] not in ".!?":
        text += "."

    dummy_audio_fp = os.path.join(OUT_DIR, "dummy_audio.wav")
    sf.write(dummy_audio_fp, np.zeros(22050 * 3), 22050)  # 3 seconds of silence
    record = {
        'audio_filepath' : dummy_audio_fp,
        'duration': 3.0,
        'text': text,
        'speaker': "dummy",
    }
    if context_text is not None:
        assert context_audio_filepath is None
        record['context_text'] = context_text
    else:
        assert context_audio_filepath is not None
        record['context_audio_filepath'] = context_audio_filepath
        record['context_audio_duration'] = get_audio_duration(context_audio_filepath)

    return record

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment Evaluation')
    parser.add_argument('--checkpoint_file', type=str, default=None, help="Path to magpie ckpt")
    parser.add_argument('--hparams_file', type=str, default=None, help="Path to magpie yaml")
    parser.add_argument('--codecmodel_path', type=str, default=None, help="Path to codec model")
    parser.add_argument('--out_dir', type=str, help="Path to store output audios")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_json', type=str, help="Path to input podcast json")
    args = parser.parse_args()

    HPARAMS_FILE = args.hparams_file
    CHECKPOINT_FILE = args.checkpoint_file
    CODECMODEL_PATH = args.codecmodel_path
    OUT_DIR = args.out_dir
    BATCH_SIZE = args.batch_size # Set this 1 for "audio" conditioning.
    CONDITIONING = "text_context" # "audio" or "text_context"
    assert CONDITIONING in ["audio", "text_context"]

    TURN_PADDING_DURATION_MS = 50 # 50ms silence in between turns
    LEGACY_TEXT_CONDITIONING = True

    # PODCAST_SCRIPT = [
    #     ["Emma", "Welcome back to the show everyone! Today we're diving into the exciting breakthroughs in AI over the past year."],
    #     ["Sean", "Yeah, it feels like every week there's a new paper or demo that blows people's minds."],
    #     ["Megan", "Absolutely. One of the biggest shifts has been the move toward multimodal AI systems that can understand text, images, and even audio together."],
    #     ["Sean", "Right, like models that can take a photo and not only describe it but also answer reasoning questions about what's happening in the scene."],
    #     ["Tom", "And it's not just perception. Generative models are evolving fast too, with image and video generation becoming incredibly realistic."],
    #     ["Emma", "I was really impressed by the new text-to-audio systems. They're starting to generate speech with emotion and nuance, almost indistinguishable from a real person."],
    #     ["Tom", "That's true, and the advances in efficiency are just as important. Training and running these models on smaller hardware footprints is opening up access for more people."],
    #     ["Sean", "Yeah, and on the ethical side, there's been more focus on alignment and safety research to make sure these powerful systems behave responsibly."],
    #     ["Tom", "I think one of the most encouraging signs is how quickly the research community is collaborating—open-sourcing models, sharing benchmarks, and building tools."],
    #     ["Emma", "Exactly. It feels like we're entering an era where AI isn't just a research curiosity, but a core part of how we work, create, and communicate."],
    #     ["Sean", "And with breakthroughs in reasoning and planning, it won't be long before AI agents can assist in much more complex tasks."],
    #     ["Tom", "Which makes the next few years both exciting and critical. We'll need to balance innovation with thoughtful deployment."]
    # ]

    with open(args.input_json, "r") as f:
        data = json.load(f)

    PODCAST_SCRIPT = []
    for turn in data["dialogue"]:
        PODCAST_SCRIPT.append([turn["speaker"], turn["text"]])

    # SPEAKER_NAME_TO_TEXTCONTEXT = {
    #     "Emma": "Speaker and Emotion: | Language:en Dataset:rivaEmmaMeganSeanTom Speaker:Emma_Conversational |",
    #     "Megan": "Speaker and Emotion: | Language:en Dataset:rivaEmmaMeganSeanTom Speaker:Megan_Conversational |",
    #     "Sean": "Speaker and Emotion: | Language:en Dataset:rivaEmmaMeganSeanTom Speaker:Sean_Conversational |",
    #     "Tom": "Speaker and Emotion: | Language:en Dataset:rivaEmmaMeganSeanTom Speaker:Tom_Conversational |",
    # }
    SPEAKER_NAME_TO_TEXTCONTEXT = {
        "Emma": "Speaker and Emotion: | Language:en Dataset:rivaEmmaMeganSeanTom Speaker:Emma_Additional |",
        "Megan": "Speaker and Emotion: | Language:en Dataset:rivaEmmaMeganSeanTom Speaker:Megan_Additional |",
        "Tom": "Speaker and Emotion: | Language:en Dataset:rivaEmmaMeganSeanTom Speaker:Tom_Additional |",
        "Sean": "Speaker and Emotion: | Language:en Dataset:rivaEmmaMeganSeanTom Speaker:Sean_Additional |",
    }



    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    model_cfg = OmegaConf.load(HPARAMS_FILE).cfg
    with open_dict(model_cfg):
        model_cfg = update_config(model_cfg, CODECMODEL_PATH, LEGACY_TEXT_CONDITIONING)

    model = MagpieTTSModel(cfg=model_cfg)
    print("Loading weights from checkpoint")
    ckpt = torch.load(CHECKPOINT_FILE, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    print("Loaded weights.")
    model.use_kv_cache_for_inference = True
    model.cuda()
    model.eval()

    text_normalizer = Normalizer(input_case='cased', lang='en')

    context_duration_min = model.cfg.get('context_duration_min', 5.0)
    context_duration_max = model.cfg.get('context_duration_max', 5.0)
    test_dataset = MagpieTTSDataset(
        dataset_meta={},
        sample_rate=model.sample_rate,
        min_duration=0.5,
        max_duration=20,
        codec_model_samples_per_frame=model.codec_model_samples_per_frame,
        bos_id=model.bos_id,
        eos_id=model.eos_id,
        context_audio_bos_id=model.context_audio_bos_id,
        context_audio_eos_id=model.context_audio_eos_id,
        audio_bos_id=model.audio_bos_id,
        audio_eos_id=model.audio_eos_id,
        num_audio_codebooks=model.num_audio_codebooks,
        prior_scaling_factor=None,
        load_cached_codes_if_available=False,
        dataset_type='test',
        tokenizer_config=None,
        load_16khz_audio=model.model_type == 'single_encoder_sv_tts',
        use_text_conditioning_tokenizer=model.use_text_conditioning_encoder,
        text_conditioning_tokenizer_name=model.text_conditioning_tokenizer_name,
        pad_context_text_to_max_duration=model.pad_context_text_to_max_duration,
        context_duration_min=context_duration_min,
        context_duration_max=context_duration_max,
    )
    test_dataset.text_tokenizer = model.tokenizer

    is_podcast_complete = False
    podcast_script_idx = 0
    speaker_wise_generations = {}
    item_idx = 0
    all_audio_paths = []

    ### Create json for dataset
    while podcast_script_idx < len(PODCAST_SCRIPT):
        audio_base_dir = "/"
        test_entries = []

        for speaker, text in PODCAST_SCRIPT[podcast_script_idx:podcast_script_idx+BATCH_SIZE]:
            if (CONDITIONING == "text_context") or (speaker not in speaker_wise_generations):
                # if text conditioning, or we haven't generated a turn for this speaker yet, use text conditioning
                test_entries.append(create_record(
                    text=text,
                    context_text=SPEAKER_NAME_TO_TEXTCONTEXT[speaker],
                ))
            else:
                # if audio conditioning, use the last generated audio for this speaker
                test_entries.append(create_record(
                    text=text,
                    context_audio_filepath=speaker_wise_generations[speaker][-1],
                ))

        data_samples = []
        for entry in test_entries:
            dataset_sample = DatasetSample(
                dataset_name="sample",
                manifest_entry=entry,
                audio_dir=audio_base_dir,
                feature_dir=audio_base_dir,
                text=entry['text'],
                speaker=None,
                speaker_index=0,
                tokenizer_names=["english_phoneme"], # Change this for multilingual: "english_phoneme", "spanish_phoneme", "english_chartokenizer", "german_chartokenizer"..
            )
            data_samples.append(dataset_sample)

        test_dataset.data_samples = data_samples

        test_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            collate_fn=test_dataset.collate_fn,
            num_workers=0,
            shuffle=False
        )


        ### Run model inference
        for bidx, batch in enumerate(test_data_loader):
            print("Processing batch {} out of {}".format(bidx, len(test_data_loader)))
            model.decoder.reset_cache(use_cache=True)
            batch_cuda ={}
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch_cuda[key] = batch[key].cuda()
                else:
                    batch_cuda[key] = batch[key]
            import time
            st = time.time()

            with torch.inference_mode():
                predicted_audio, predicted_audio_lens, _, _, rtf_metrics, cross_attn_np, _ = model.infer_batch(
                    batch_cuda,
                    max_decoder_steps=430,
                    temperature=0.7,
                    topk=80,
                    use_cfg=True,
                    cfg_scale=2.5,
                    prior_epsilon=0.2,
                    lookahead_window_size=5,
                    return_cross_attn_probs=True,
                    estimate_alignment_from_layers=[6,7,8],
                    apply_attention_prior=True,
                    apply_prior_to_layers=[4,5,6,7,8,9,10,11],
                    compute_all_heads_attn_maps=False,  # always off
                    start_prior_after_n_audio_steps=0,
                    use_local_transformer_for_inference=True,
                    ignore_finished_sentence_tracking=True,
                    eos_detection_method="argmax_or_multinomial_any"
                )

            for idx in range(predicted_audio.size(0)):
                predicted_audio_np = predicted_audio[idx].float().detach().cpu().numpy()
                predicted_audio_np = predicted_audio_np[:predicted_audio_lens[idx]]
                audio_path = os.path.join(OUT_DIR, f"predicted_audio_{item_idx}.wav")
                sf.write(audio_path, predicted_audio_np, model.sample_rate)
                all_audio_paths.append(audio_path)
                speaker = PODCAST_SCRIPT[item_idx][0]
                if speaker not in speaker_wise_generations:
                    speaker_wise_generations[speaker] = []
                speaker_wise_generations[speaker].append(audio_path)
                item_idx += 1

            # Combine all audio files into a single audio file with a 50ms silence in between
            audio_list = []
            for audio_path in all_audio_paths:
                audio_list.append(sf.read(audio_path)[0])
                silent_audio = np.zeros(int(model.sample_rate * TURN_PADDING_DURATION_MS/1000))
                audio_list.append(silent_audio)
            combined_audio = np.concatenate(audio_list)
            combined_audio_path = os.path.join(OUT_DIR, "combined_audio.wav")
            sf.write(combined_audio_path, combined_audio, model.sample_rate)
            print(f"Saved combined audio to: {combined_audio_path}")

        podcast_script_idx += len(test_entries)

    print("Podcast complete!")