# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import tensorflow as tf
import nkutil
from text import yinsu_symbols, character_symbols, pinyin_symbols


def create_hparams():
    hparams =  nkutil.HParams(
        ################################
        # PolyPhonic Parameters        #
        ################################
        num_classes = 1665,    ##？
        class2idx = "./filelists/uni_class2idx.json",
        merge_cedict = "./filelists/universal_cedict.json",
        saved_model_path_poly = "./save/poly_only/97.98_model.pt",
        saved_model_path_structure_poly = "./save/poly_only_syntax_frozen/97.16_model.pt",

        train_file = "./filelists/train_polyphonic.sent",
        train_label = "./filelists/train_polyphonic.lb",
        val_file = "./filelists/dev_polyphonic.sent",
        val_label = "./filelists/dev_polyphonic.lb",
        test_file = "./filelists/test_polyphonic.sent",
        test_label = "./filelists/test_polyphonic.lb",

        poly_batch_size = 32,
        poly_max_length = 512,
        poly_epochs = 1500,
        poly_lr = 5e-5,
        use_output_mask = True,

        # control whether use syntax structure information in TTS
        poly_use_structure = True,
        tts_use_structure = True,
        encoder_input_dim=[812, 512, 512],
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=500,
        seed=4321,   #?
        dynamic_loss_scaling=True,
        fp16_run=False,   ##?
        distributed_run=False,   ##?
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",  ## str 这个URL指定了如何初始化互相通信的进程。
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],


        synth_batch_size = 1,
        ################################
        # Data Parameters             #
        ################################
        # load_mel_from_disk=False,
        load_mel_from_disk=True,     ##? 人工切换
        pretrain_model_path_structure = './models/ch_bert_bmes_dev=93.97.pt',
        saved_model_path_sandhi_structure = './save/poly_tts_CNN_syntax_frozen/96.39_model.pt',
        saved_model_path_sandhi = './save/poly_tts_CNN/96.84_model.pt',
        training_files='filelists/bznsyp_character_audio_text_train_filelist.txt',
        validation_files='filelists/bznsyp_character_audio_text_val_filelist.txt',
        mel_training_files='filelists/mel-bznsyp_character_audio_text_train_filelist.txt',
        mel_validation_files='filelists/mel-bznsyp_character_audio_text_val_filelist.txt',
        polyphone_dict_files = 'filelists/polyphone_dict.json',
        mask_dict_files = 'filelists/polyphone_mask.json',
        text_cleaners=['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=16000,
        filter_length=1024,
        # hop_length=256,
        # win_length=1024,
        hop_length = 200,
        win_length = 800,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        
        signal_normalization = True,
        allow_clipping_in_normalization = True,
        use_lws=False,
        max_abs_value = 4.,
        symmetric_mels = True,
        min_level_db = -100,
        ref_level_db = 20,
        magnitude_power = 2.,
        fmin = 55,
        fmax = 7600,
        power = 1.5,
        griffin_lim_iters = 60,
        preemphasize = True,
        preemphasis = 0.97,

        ################################
        # Model Parameters             #
        ################################
        n_yinsu_symbols=len(yinsu_symbols),
        n_character_symbols=len(character_symbols),
        n_pinyin_symbols=1665,
        character_symbols_embedding_dim=512,
        yinsu_symbols_embedding_dim=512,
        structure_feature_dim=300,
        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        # encoder_embedding_dim=1836,  # = 1024 + 512 + 300
        # encoder_embedding_dim=1324,  # = 1024 + 300
        encoder_embedding_dim=512,     # 512
        encoder_output_dim=[512, 512, 512],

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-4,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=32,
        mask_padding=True,  # set model's padded outputs to padded values
    )
    return hparams

    # hparams = {
        # "epochs": 500,
        # "iters_per_checkpoint": 500,
        # "seed": 1234,
        # "dynamic_loss_scaling": True,
        # "fp16_run": False,
        # "distributed_run": False,
        # "dist_backend": "nccl",
        # "dist_url": 'tcp://localhost:54321',
        # "cudnn_enabled": True,
        # "cudnn_benchmark": False,
        # "ignore_layers": ['embedding.weight'],

        # "load_mel_from_disk": False,
        # "training_files": 'filelists/bznsyp_character_audio_text_train_filelist.txt',
        # "validation_files": 'filelists/bznsyp_character_audio_text_val_filelist.txt',
        # "polyphone_dict_files": 'filelists/polyphone_dict.json',
        # "mask_dict_files": 'filelists/polyphone_mask.json',
        # "text_cleaners": ['english_cleaners'],

        # "max_wav_value": 32768.0,
        # "sampling_rate": 16000,
        # "filter_length": 1024,
        # # "hop_length": 256,
        # # "win_length": 1024,
        # "hop_length": 200,
        # "win_length": 800,
        # "n_mel_channels": 80,
        # "mel_fmin": 0.0,
        # "mel_fmax": 8000.0,

        # "n_yinsu_symbols": len(yinsu_symbols),
        # "n_character_symbols": len(character_symbols),
        # "n_pinyin_symbols": len(pinyin_symbols),
        # "character_symbols_embedding_dim": 512,
        # "yinsu_symbols_embedding_dim": 512,

        # "encoder_kernel_size": 5,
        # "encoder_n_convolutions": 3,
        # "encoder_embedding_dim": 512,

        # "n_frames_per_step": 1,  # currently only 1 is supported
        # "decoder_rnn_dim": 1024,
        # "prenet_dim": 256,
        # "max_decoder_steps": 1000,
        # "gate_threshold": 0.5,
        # "p_attention_dropout": 0.1,
        # "p_decoder_dropout": 0.1,

        # "attention_rnn_dim": 1024,
        # "attention_dim": 128,

        # "attention_location_n_filters": 32,
        # "attention_location_kernel_size": 31,

        # "postnet_embedding_dim": 512,
        # "postnet_kernel_size": 5,
        # "postnet_n_convolutions": 5,

        # "use_saved_learning_rate": False,
        # "learning_rate": 1e-3,
        # "weight_decay": 1e-6,
        # "grad_clip_thresh": 1.0,
        # "batch_size": 8,
        # "mask_padding": True  
    # }


    # hparams = tf.contrib.training.HParams(
        # ################################
        # # Experiment Parameters        #
        # ################################
        # epochs=500,
        # iters_per_checkpoint=500,
        # seed=1234,
        # dynamic_loss_scaling=True,
        # fp16_run=False,
        # distributed_run=False,
        # dist_backend="nccl",
        # dist_url="tcp://localhost:54321",
        # cudnn_enabled=True,
        # cudnn_benchmark=False,
        # ignore_layers=['embedding.weight'],

        # ################################
        # # Data Parameters             #
        # ################################
        # load_mel_from_disk=False,
        # training_files='filelists/bznsyp_character_audio_text_train_filelist.txt',
        # validation_files='filelists/bznsyp_character_audio_text_val_filelist.txt',
        # polyphone_dict_files = 'filelists/polyphone_dict.json',
        # mask_dict_files = 'filelists/polyphone_mask.json',
        # text_cleaners=['english_cleaners'],

        # ################################
        # # Audio Parameters             #
        # ################################
        # max_wav_value=32768.0,
        # sampling_rate=16000,
        # filter_length=1024,
        # # hop_length=256,
        # # win_length=1024,
        # hop_length = 200,
        # win_length = 800,
        # n_mel_channels=80,
        # mel_fmin=0.0,
        # mel_fmax=8000.0,

        # ################################
        # # Model Parameters             #
        # ################################
        # n_yinsu_symbols=len(yinsu_symbols),
        # n_character_symbols=len(character_symbols),
        # n_pinyin_symbols=len(pinyin_symbols),
        # character_symbols_embedding_dim=512,
        # yinsu_symbols_embedding_dim=512,

        # # Encoder parameters
        # encoder_kernel_size=5,
        # encoder_n_convolutions=3,
        # # encoder_embedding_dim=1836,  # = 1024 + 512 + 300
        # # encoder_embedding_dim=1324,  # = 1024 + 300
        # encoder_embedding_dim=512,  # = 1024 + 300 + 512

        # # Decoder parameters
        # n_frames_per_step=1,  # currently only 1 is supported
        # decoder_rnn_dim=1024,
        # prenet_dim=256,
        # max_decoder_steps=1000,
        # gate_threshold=0.5,
        # p_attention_dropout=0.1,
        # p_decoder_dropout=0.1,

        # # Attention parameters
        # attention_rnn_dim=1024,
        # attention_dim=128,

        # # Location Layer parameters
        # attention_location_n_filters=32,
        # attention_location_kernel_size=31,

        # # Mel-post processing network parameters
        # postnet_embedding_dim=512,
        # postnet_kernel_size=5,
        # postnet_n_convolutions=5,

        # ################################
        # # Optimization Hyperparameters #
        # ################################
        # use_saved_learning_rate=False,
        # learning_rate=1e-3,
        # weight_decay=1e-6,
        # grad_clip_thresh=1.0,
        # batch_size=8,
        # mask_padding=True  # set model's padded outputs to padded values
    # )