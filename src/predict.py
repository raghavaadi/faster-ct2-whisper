"""
This file contains the Predictor class, which is used to run predictions on the
Whisper model. It is based on the Predictor class from the original Whisper
repository, with some modifications to make it work with the RP platform.
"""

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging

from runpod.serverless.utils import rp_cuda

from faster_whisper import WhisperModel
from faster_whisper.utils import format_timestamp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Predictor:
    """ A Predictor class for the Whisper model """

    def __init__(self):
        self.models = {}

    def load_model(self, model_name):
        """ Load the model from the weights folder. """
        logger.info(f"Loading model: {model_name}")
        loaded_model = WhisperModel(
            model_name,
            device="cuda" if rp_cuda.is_available() else "cpu",
            compute_type="float16" if rp_cuda.is_available() else "int8")
        logger.info(f"Model {model_name} loaded successfully")
        return model_name, loaded_model

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        logger.info("Setting up models")
        model_names = ["tiny", "base", "large-v3","large-v3-turbo-ct2"]
        with ThreadPoolExecutor() as executor:
            for model_name, model in executor.map(self.load_model, model_names):
                if model_name is not None:
                    self.models[model_name] = model
        logger.info("All models loaded successfully")

    def predict(
        self,
        audio,
        model_name="base",
        transcription="plain_text",
        translate=False,
        translation="plain_text",
        language=None,
        temperature=0,
        best_of=5,
        beam_size=5,
        patience=1,
        length_penalty=None,
        suppress_tokens="-1",
        initial_prompt=None,
        condition_on_previous_text=True,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        enable_vad=False,
        word_timestamps=False
    ):
        """
        Run a single prediction on the model
        """
        logger.info(f"Starting prediction with model: {model_name}")
        model = self.models.get(model_name)
        if not model:
            logger.error(f"Model '{model_name}' not found")
            raise ValueError(f"Model '{model_name}' not found.")

        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        logger.info("Starting transcription")
        segments, info = list(model.transcribe(str(audio),
                                               language=language,
                                               task="transcribe",
                                               beam_size=beam_size,
                                               best_of=best_of,
                                               patience=patience,
                                               length_penalty=length_penalty,
                                               temperature=temperature,
                                               compression_ratio_threshold=compression_ratio_threshold,
                                               log_prob_threshold=logprob_threshold,
                                               no_speech_threshold=no_speech_threshold,
                                               condition_on_previous_text=condition_on_previous_text,
                                               initial_prompt=initial_prompt,
                                               prefix=None,
                                               suppress_blank=True,
                                               suppress_tokens=[-1],
                                               without_timestamps=False,
                                               max_initial_timestamp=1.0,
                                               word_timestamps=word_timestamps,
                                               vad_filter=enable_vad
                                               ))

        segments = list(segments)
        logger.info(f"Transcription completed. Detected language: {info.language}")

        transcription = format_segments(transcription, segments)

        results = {
            "segments": serialize_segments(segments),
            "detected_language": info.language,
            "transcription": transcription,
            "device": "cuda" if rp_cuda.is_available() else "cpu",
            "model": model_name,
        }

        if translate:
            logger.info("Starting translation")
            translation_segments, translation_info = model.transcribe(
                str(audio),
                task="translate",
                temperature=temperature,
                word_timestamps=word_timestamps
            )
            translation_segments = list(translation_segments)
            logger.info("Translation completed")

            translation = format_segments(translation, translation_segments)
            results["translation"] = translation
            results["translated_segments"] = serialize_segments(translation_segments)

        if word_timestamps:
            logger.info("Processing word timestamps")
            word_timestamps = []
            segments_to_use = translation_segments if translate else segments
            for segment in segments_to_use:
                for word in segment.words:
                    word_timestamps.append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                    })
            results["word_timestamps"] = word_timestamps
            logger.info(f"Processed {len(word_timestamps)} word timestamps")

        logger.info("Prediction completed successfully")
        return results


def serialize_segments(transcript):
    '''
    Serialize the segments to be returned in the API response.
    '''
    logger.info("Serializing segments")
    serialized = [{
        "id": segment.id,
        "seek": segment.seek,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "tokens": segment.tokens,
        "temperature": segment.temperature,
        "avg_logprob": segment.avg_logprob,
        "compression_ratio": segment.compression_ratio,
        "no_speech_prob": segment.no_speech_prob
    } for segment in transcript]
    logger.info(f"Serialized {len(serialized)} segments")
    return serialized


def format_segments(format, segments):
    '''
    Format the segments to the desired format
    '''
    logger.info(f"Formatting segments to {format}")

    if format == "plain_text":
        result = " ".join([segment.text.lstrip() for segment in segments])
    elif format == "formatted_text":
        result = "\n".join([segment.text.lstrip() for segment in segments])
    elif format == "srt":
        result = write_srt(segments)
    else:
        result = write_vtt(segments)

    logger.info("Segment formatting completed")
    return result


def write_vtt(transcript):
    '''
    Write the transcript in VTT format.
    '''
    logger.info("Writing transcript in VTT format")
    result = ""

    for segment in transcript:
        result += f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    logger.info("VTT formatting completed")
    return result


def write_srt(transcript):
    '''
    Write the transcript in SRT format.
    '''
    logger.info("Writing transcript in SRT format")
    result = ""

    for i, segment in enumerate(transcript, start=1):
        result += f"{i}\n"
        result += f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
        result += f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    logger.info("SRT formatting completed")
    return result
