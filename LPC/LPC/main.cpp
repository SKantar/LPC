#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <vector>

#define BITS_PER_BYTE 8
#define ENERGY_OF_SAMPLE true
#define VALUE_OF_SAMPLE false
#define WORK_SIZE_ALGORITHM_D 100
#define WINDOW_SIZE_ALGORITHM_D 10
#define ALLOWED_ERROR 1e-05
#define D_VAR_X 3
#define PI 3.141592653589793238462643383279502884L

using namespace std;

// WAVE PCM soundfile format (you can find more in https://ccrma.stanford.edu/courses/422/projects/WaveFormat/ )
typedef struct header_file{
	char chunk_id[4];
	int chunk_size;
	char format[4];
	char subchunk1_id[4];
	int subchunk1_size;
	short int audio_format;
	short int num_channels;
	int sample_rate;			// sample_rate denotes the sampling rate.
	int byte_rate;
	short int block_align;
	short int bits_per_sample;
	char subchunk2_id[4];
	int subchunk2_size;			// subchunk2_size denotes the number of samples.
} header;

typedef struct frame{
	double *samples;
	double *features;
	int number_of_samples;
}frame;

typedef struct word_bounds{
	int start;
	int end;
}word_bounds;

typedef struct word{
	frame* *frames;
	int number_of_frames;
}word;


int ms_to_samples(int ms, int sample_rate){
	return (sample_rate * ms) / 1000;
}

int get_samples(FILE* file, short int* samples, int number_of_samples, bool energy){
	short int left_chanel, right_chanel;
	int number_of_read = 0;
	for (int i = 0; i < number_of_samples; i++){
		fread(&left_chanel, 1, sizeof(short int), file);
		fread(&right_chanel, 1, sizeof(short int), file);

		if (energy){
			left_chanel = abs(left_chanel);
			right_chanel = abs(right_chanel);
		}
		//DA LI JE OVO VAKO?! :D
		//if (energy){
			samples[i] = max(left_chanel, right_chanel);
		//}else{
		//Ovo je trenutno
		//samples[i] = left_chanel;
		//}
		number_of_read++;
		if (feof(file)) break;
	}
	return number_of_read;
}

double arithmetic_mean(short int* arr, int n){
	int result = 0;
	for (int i = 0; i < n; i++)
		result += arr[i];
	return (double)result / n;
}

double standard_deviation(short int *samples, int number_of_samples){
	double deviation = 0;
	double mean = arithmetic_mean(samples, number_of_samples);
	for (int i = 0; i < number_of_samples; i++)
		deviation += (samples[i] - mean)*(samples[i] - mean);
	deviation /= number_of_samples;
	return sqrt(deviation);
}


double treshold_of_noise(FILE* file, int sample_rate){
	int number_of_samples = ms_to_samples(WORK_SIZE_ALGORITHM_D, sample_rate);
	short int* samples = (short int*)malloc(number_of_samples * sizeof(short int));
	int number_of_read = get_samples(file, samples, number_of_samples, ENERGY_OF_SAMPLE);
	double deviation = standard_deviation(samples, number_of_samples);
	double mean = arithmetic_mean(samples, number_of_samples);
	return mean + 2 * deviation;
}

void deleteWord(int ind, vector<word_bounds*> &words){
	words.erase(words.begin() + ind);
	//cout << "desilo se" <<words.size() <<  endl;

}


void extra_smoothing(int length_of_wav, vector<word_bounds*> &words){
	int n = words.size();
	int size = 70;
	// prvo provera "laznih" tisina
	for (int i = 0; i<n - 1; i++){
		if (words[i + 1]->start - words[i]->end <= size &&
			words[i + 1]->end - words[i]->start >= 300){
			// tisina u govor
			words[i]->end = words[i + 1]->end;
			deleteWord(i + 1, words);
			i--;
			n--;
		}
	}

	n = words.size();
	// prva rec
	// pauza pre prvi reci 100MS
	if (n>1 && words[0]->end - words[0]->start <= size &&
		words[1]->start >= 300){
		deleteWord(0, words);
		n--;
	}

	// drugo provera "laznih" govora
	for (int i = 0; i<n - 1; i++){
		if (words[i]->end - words[i]->start <= size &&
			words[i + 1]->start - words[i - 1]->end >= 300){
			deleteWord(i, words);
			i--;
			n--;
		}
	}

	//poslednja rec
	if (n>1 && words[n - 1]->end - words[n - 1]->start <= size &&
		length_of_wav - words[n - 2]->end >= 300){
		deleteWord(n - 1, words);
	}

	//cout << "Ovde " << words.size() << endl;

}

vector<word_bounds*> algorithm_D(FILE* file, int sample_rate){
	bool speech_state = false;
	int offset = 100;
	int number_of_good_windows = 0;
	int number_of_read;
	vector<word_bounds*> words;
	word_bounds* temp_word;
	double noise = treshold_of_noise(file, sample_rate);
	int number_of_samples_in_window = ms_to_samples(WINDOW_SIZE_ALGORITHM_D, sample_rate);
	while (true){
		short int* samples = (short int*)malloc(number_of_samples_in_window * sizeof(short int));
		number_of_read = get_samples(file, samples, number_of_samples_in_window, ENERGY_OF_SAMPLE);		//procitati prozor
		double mean = arithmetic_mean(samples, number_of_samples_in_window);

		if (mean - noise >= ALLOWED_ERROR){
			if (number_of_good_windows < 0){
				number_of_good_windows = 0;
			}
			else{
				number_of_good_windows++;
				if (number_of_good_windows >= D_VAR_X)
					if (!speech_state){
					speech_state = true;
					temp_word = (word_bounds*)malloc(sizeof(word_bounds));
					temp_word->start = offset - D_VAR_X * WINDOW_SIZE_ALGORITHM_D;
					//wordsBegin.push_back(offset - D_VAR_X * WINDOW_SIZE_ALGORITHM_D);
					}
			}
		}
		else {
			if (number_of_good_windows > 0){
				number_of_good_windows = 0;
			}
			else{
				number_of_good_windows--;
				if (number_of_good_windows <= -D_VAR_X)
					if (speech_state){
					speech_state = false;
					temp_word->end = offset - D_VAR_X * WINDOW_SIZE_ALGORITHM_D;
					words.push_back(temp_word);
					//free(temp_word);
					//wordsEnd.push_back(offset - D_VAR_X * WINDOW_SIZE_ALGORITHM_D);
					}
			}
		}

		offset += WINDOW_SIZE_ALGORITHM_D;
		if (number_of_read < number_of_samples_in_window) break;
	}

	if (speech_state){
		speech_state = false;
		//wordsEnd.push_back(offset);
		temp_word->end = offset;
		words.push_back(temp_word);
		//free(temp_word);
	}
	int length_of_wav = offset;
	extra_smoothing(length_of_wav, words);
	return words;
}

int get_word_size(FILE* file, word_bounds* bounds, int sample_rate){
	int sample_start = ms_to_samples(bounds->start, sample_rate);
	int sample_end = ms_to_samples(bounds->end, sample_rate);

	int word_size = sample_end - sample_start + 1;
	return word_size;
}

double* read_word(FILE* file, word_bounds* bounds, int sample_rate){
	int sample_start = ms_to_samples(bounds->start, sample_rate);
	int sample_end = ms_to_samples(bounds->end, sample_rate);

	int word_size = sample_end - sample_start + 1;
	short int* temp_sample = (short int*)malloc(1 * sizeof(short int));
	double* samples = (double*)malloc(sizeof(double)*word_size);
	fseek(file, sizeof(header) + 4 * word_size, SEEK_SET);

	for (int i = sample_start; i <= sample_end; i++) {
		int number_of_read = get_samples(file, temp_sample, 1, false);
		samples[i - sample_start] = temp_sample[0];
		//samples[i - sample_start] = readNSamples(1, 0)[0];
	}

	return samples;
}

frame* create_frame(int number_of_samples, int feature_size) {
	frame* f = (frame*)malloc(sizeof(frame));
	f->samples = (double*)malloc(sizeof(double) * number_of_samples);
	f->features = (double*)malloc(sizeof(double) * feature_size);
	f->number_of_samples = number_of_samples;
	return f;
}

double hamming(int index, int frame_length) {
	return  0.53836 - 0.46164 * cos(2 * PI * index / (frame_length - 1));
}

int get_number_of_frames(int samples_size, int frame_length, int sample_rate){
	int samples_per_frame = ms_to_samples(frame_length, sample_rate);
	double number_of_frames = ceil(samples_size / samples_per_frame);
	return (int)number_of_frames;
}

frame** make_frames(double *samples, int samples_size, int frame_length, int sample_rate, int feature_size){
	int samples_per_frame = ms_to_samples(frame_length, sample_rate);
	int number_of_frames = get_number_of_frames(samples_size, frame_length, sample_rate);
	frame** frames = (frame**)malloc(sizeof(frame*) * number_of_frames);
	frame* f = create_frame(samples_per_frame, feature_size);
	int temp = 0;
	int pos_in_result = 0;

	for (int i = 0; i < samples_size; i++){
		if (temp == samples_per_frame){
			frames[pos_in_result++] = f;
			f = create_frame(samples_per_frame, feature_size);
			temp = 0;
		}
		f->samples[temp] = samples[i];
		temp++;
	}


	//epic math
	for (int i = 0; i < number_of_frames; i++){
		int k = frames[i]->number_of_samples;
		for (int j = 0; j < k; j++){
			frames[i]->samples[j] = frames[i]->samples[j] * hamming(j, k);
		}
	}

	return frames;
}

double* calculate_correlation_values(frame* f, int size) {
	double* correlation_values = (double*)malloc(sizeof(double)*(size + 1));

	for (int i = 0; i <= size; i++) {
		correlation_values[i] = 0;
		for (int j = 0; j < f->number_of_samples; j++) {
			correlation_values[i] += f->samples[j] * (j - i < 0 ? 0 : f->samples[j - i]);
		}
	}
	return correlation_values;
}


double** forward_substitution(double **coefficient_matrix, int size){
	int max;
	double t;
	for (int i = 0; i < size; i++) {
		max = i;
		for (int j = i + 1; j < size; j++)
			if (coefficient_matrix[j][i] > coefficient_matrix[max][i])
				max = j;

		for (int j = 0; j < size + 1; j++) {
			t = coefficient_matrix[max][j];
			coefficient_matrix[max][j] = coefficient_matrix[i][j];
			coefficient_matrix[i][j] = t;
		}

		for (int j = size; j >= i; j--)
			for (int k = i + 1; k < size; k++)
				coefficient_matrix[k][j] -= coefficient_matrix[k][i] / coefficient_matrix[i][i] * coefficient_matrix[i][j];
	}
	return coefficient_matrix;
}

double** reverse_elimination(double **coefficient_matrix, int size){
	for (int i = size - 1; i >= 0; i--) {
		coefficient_matrix[i][size] = coefficient_matrix[i][size] / coefficient_matrix[i][i];
		coefficient_matrix[i][i] = 1;
		for (int j = i - 1; j >= 0; j--) {
			coefficient_matrix[j][size] -= coefficient_matrix[j][i] * coefficient_matrix[i][size];
			coefficient_matrix[j][i] = 0;
		}
	}
	return coefficient_matrix;
}


double** gauss(double **coefficient_matrix, int size) {
	coefficient_matrix = forward_substitution(coefficient_matrix, size);
	return reverse_elimination(coefficient_matrix, size);
}


double* calculate_lpc_coefficients(double *correlation_values, int size){

	double **coefficient_matrix = (double **)malloc(size * sizeof(double*));
	for (int i = 0; i < size; i++) coefficient_matrix[i] = (double*)malloc((size + 1) * sizeof(double));

	for (int i = 0; i < size; ++i)
		for (int j = 0; j < size; ++j)
			coefficient_matrix[i][j] = correlation_values[abs(i - j)];

	for (int i = 0; i < size; i++)
		coefficient_matrix[i][size] = correlation_values[i + 1];

	coefficient_matrix = gauss(coefficient_matrix, size);

	double* features = (double*)malloc(sizeof(double)*size);

	for (int i = 0; i < size; i++)
		features[i] = coefficient_matrix[i][size];

	return features;

}


//frame_length in MS
word* create_word(double *samples, int samples_size, int frame_length, int sample_rate, int feature_size){
	
	int number_of_frames = get_number_of_frames(samples_size, frame_length, sample_rate);
	frame** frames = make_frames(samples, samples_size, frame_length, sample_rate, feature_size);
	// calculate correlation values && lpc values for every frame
	//JUST DO RIGHT THING :D
	for (int i = 0; i < number_of_frames; i++){
		double* correlation_values = calculate_correlation_values(frames[i], feature_size);
		frames[i]->features = calculate_lpc_coefficients(correlation_values, feature_size);
	}
	word *w = (word *)malloc(sizeof(word));
	w->frames = frames;
	w->number_of_frames = number_of_frames;
	return w;
}

double euclidian_distance(double* array_1, double* array_2, int size_1, int size_2) {
	double distance = 0;
	for (int i = 0; i < max(size_1, size_2); i++) {
		distance += ((i < size_1 ? array_1[i] : 0) - (i < size_2 ? array_2[i] : 0))*((i < size_1 ? array_1[i] : 0) - (i < size_2 ? array_2[i] : 0));
	}
	return sqrt(distance);
}



int main(){
	FILE * infile;
	errno_t err;
	int no_of_samples;
	int size_of_sample;
	int features_size_param1 = 12; 
	int frame_length_param1 = 20; //ms
	int features_size_param2 = 13;
	int frame_length_param2 = 30; //ms
	int features_size_param3 = 14;
	int frame_length_param3 = 40; //ms
	int features_size_param4 = 15;
	int frame_length_param4 = 10; //ms


	err = fopen_s(&infile, "Bfight.wav", "rb");											// short int used for 16 bit as input data format is 16 bit PCM audio
	header* meta = (header*)malloc(sizeof(header));											// header_p points to a header struct that contains the wave file metadata fields

	//freopen("Bfight.txt", "w", stdout);

	if (!err){
		//header
		fread(meta, 1, sizeof(header), infile);
		vector<word_bounds*> bounds = algorithm_D(infile, meta->sample_rate);
		//cout << sizeof(header);
		//about samples
		no_of_samples = meta->subchunk2_size / meta->block_align;
		size_of_sample = meta->bits_per_sample / BITS_PER_BYTE;
		
		int word_size = get_word_size(infile, bounds[0], meta->sample_rate);
		double* samples = read_word(infile, bounds[0], meta->sample_rate);

		word* w1 = create_word(samples, word_size, frame_length_param1, meta->sample_rate, features_size_param1);
		word* w2 = create_word(samples, word_size, frame_length_param2, meta->sample_rate, features_size_param2);
		word* w3 = create_word(samples, word_size, frame_length_param3, meta->sample_rate, features_size_param3);
		word* w4 = create_word(samples, word_size, frame_length_param4, meta->sample_rate, features_size_param4);


		//cout << w->frames[0]->number_of_samples << endl;
	/*	cout <<endl<< "1 - 2" << endl;
		for (int i = 0; i < w1->number_of_frames; i++){
			cout << euclidian_distance(w1->frames[i]->features, w2->frames[i]->features, features_size_param1, features_size_param2) << " ";
		}
		cout <<endl<< "1 - 3" << endl;
		for (int i = 0; i < w1->number_of_frames; i++){
			cout << euclidian_distance(w1->frames[i]->features, w3->frames[i]->features, features_size_param1, features_size_param3) << " ";
		}
		cout <<endl<< "1 - 4" << endl;
		for (int i = 0; i < w1->number_of_frames; i++){
			cout << euclidian_distance(w1->frames[i]->features, w4->frames[i]->features, features_size_param1, features_size_param4) << " ";
		}*/

		/*for (int i = 0; i < word_size; i++)
			cout << samples[i] << " ";
		cout << endl;*/
		
		//int size = read_word(infile, words[0], samples, meta->sample_rate);
		
		cout << "Velicina: " << bounds.size() << endl;
		cout << w1->number_of_frames << endl;
		cout << w2->number_of_frames << endl;
		cout << w3->number_of_frames << endl;
		cout << w4->number_of_frames << endl;

		//fclose(infile);
		/*cout << "\t" << bounds[0]->start << " - " << bounds[0]->end << endl;
		cout << bounds[0]->end - bounds[0]->start << endl;
		cout << meta->sample_rate << endl;
		cout << ms_to_samples(bounds[0]->end - bounds[0]->start + 1, 48000) << endl;
		for (int i = 0; i < word_size; i++)
		cout << samples[i] << " ";
		cout << endl;*/
		//cout << "\t" << words[1]->start << " - " << words[1]->end << endl;
		//cout << "\t" << words[2]->start << " - " << words[2]->end << endl;
		//writeHeader(meta);
	}

	return 0;
}