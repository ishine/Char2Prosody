import argparse
import os
from multiprocessing import cpu_count

from text import preprocessor
# from hparams import hparams
from hparams import create_hparams
from tqdm import tqdm

train_size = 9950
val_size = 15

from text import pinyin_symbols

def preprocess(args, input_folders, out_dir, hparams):
	mel_dir = os.path.join(out_dir, 'mels')
	wav_dir = os.path.join(out_dir, 'audio')
	linear_dir = os.path.join(out_dir, 'linear')
	pitch_dir = os.path.join(out_dir, 'pitch')
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(linear_dir, exist_ok=True)
	os.makedirs(pitch_dir, exist_ok=True)
	# metadata = preprocessor.build_from_path(hparams, input_folders, mel_dir, linear_dir, pitch_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	metadata = preprocessor.build_from_path(hparams, input_folders, out_dir, mel_dir, linear_dir, pitch_dir, wav_dir, pinyin_symbols, 1, tqdm=tqdm)
	write_metadata(metadata, out_dir)

def write_metadata(metadata, out_dir):
	with open(os.path.join(out_dir, 'bznsyp_audio_text_train_filelist.txt'), 'w', encoding='utf-8') as f:
		for m in metadata[:train_size]:
			f.write('|'.join([str(x) for x in m]) + '\n')
	with open(os.path.join(out_dir, 'bznsyp_audio_text_test_filelist.txt'), 'w', encoding='utf-8') as f:
		for m in metadata[train_size:train_size+val_size]:
			f.write('|'.join([str(x) for x in m]) + '\n')
	with open(os.path.join(out_dir, 'bznsyp_audio_text_val_filelist.txt'), 'w', encoding='utf-8') as f:
		for m in metadata[train_size+val_size:]:
			f.write('|'.join([str(x) for x in m]) + '\n')
	# mel_frames = sum([int(m[4]) for m in metadata])
	# timesteps = sum([int(m[3]) for m in metadata])
	# sr = hparams.sample_rate
	# hours = timesteps / sr / 3600
	# print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
		# len(metadata), mel_frames, timesteps, hours))
	# print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
	# print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
	# print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))

def norm_data(args):

	merge_books = (args.merge_books=='True')

	print('Selecting data folders..')
	supported_datasets = ['LJSpeech-1.0', 'LJSpeech-1.1', 'M-AILABS', 'BZNSYP']
	if args.dataset not in supported_datasets:
		raise ValueError('dataset value entered {} does not belong to supported datasets: {}'.format(
			args.dataset, supported_datasets))

	if args.dataset.startswith('LJSpeech'):
		return [os.path.join(args.base_dir, args.dataset)]

	if args.dataset == 'BZNSYP':
		return ['/home/sch19/data/BZNSYP']

	if args.dataset == 'M-AILABS':
		supported_languages = ['en_US', 'en_UK', 'fr_FR', 'it_IT', 'de_DE', 'es_ES', 'ru_RU',
			'uk_UK', 'pl_PL', 'nl_NL', 'pt_PT', 'fi_FI', 'se_SE', 'tr_TR', 'ar_SA']
		if args.language not in supported_languages:
			raise ValueError('Please enter a supported language to use from M-AILABS dataset! \n{}'.format(
				supported_languages))

		supported_voices = ['female', 'male', 'mix']
		if args.voice not in supported_voices:
			raise ValueError('Please enter a supported voice option to use from M-AILABS dataset! \n{}'.format(
				supported_voices))

		path = os.path.join(args.base_dir, args.language, 'by_book', args.voice)
		supported_readers = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path,e))]
		if args.reader not in supported_readers:
			raise ValueError('Please enter a valid reader for your language and voice settings! \n{}'.format(
				supported_readers))

		path = os.path.join(path, args.reader)
		supported_books = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path,e))]
		if merge_books:
			return [os.path.join(path, book) for book in supported_books]

		else:
			if args.book not in supported_books:
				raise ValueError('Please enter a valid book for your reader settings! \n{}'.format(
					supported_books))

			return [os.path.join(path, args.book)]


def run_preprocess(args, hparams):
	input_folders = norm_data(args)
	output_folder = os.path.join(args.base_dir, args.output)

	preprocess(args, input_folders, output_folder, hparams)


def main():
	print('initializing preprocessing..')
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='/home/sch19/data/BZNSYP')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	# parser.add_argument('--dataset', default='LJSpeech-1.1')
	parser.add_argument('--dataset', default='BZNSYP')
	parser.add_argument('--language', default='en_US')
	parser.add_argument('--voice', default='female')
	parser.add_argument('--reader', default='mary_ann')
	parser.add_argument('--merge_books', default='False')
	parser.add_argument('--book', default='northandsouth')
	parser.add_argument('--output', default='trainingDate_v2')
	parser.add_argument('--n_jobs', type=int, default=cpu_count())
	args = parser.parse_args()

	# modified_hp = hparams.parse(args.hparams)
	hparams = create_hparams(args.hparams)

	assert args.merge_books in ('False', 'True')

	run_preprocess(args, hparams)


if __name__ == '__main__':
	main()
