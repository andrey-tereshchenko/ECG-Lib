import numpy as np
import matplotlib.pyplot as plt
from ecgdetectors import Detectors
from sklearn.cluster import KMeans
from gensim.models import Word2Vec


def detect_r_peaks(ecg_record, fs):
    detector = Detectors(fs)
    unfiltered_ecg = ecg_record[0][:, 1]
    r_peaks = detector.engzee_detector(unfiltered_ecg)
    return r_peaks


def select_peaks_by_ecg_size(peaks, ecg_size):
    selected_peaks = []
    for peak in peaks:
        if peak < ecg_size:
            selected_peaks.append(peak)
        else:
            break
    return selected_peaks


def draw_ecg_with_r_peak(peaks, record, ecg_size):
    draw_peaks = select_peaks_by_ecg_size(peaks, ecg_size)
    plt.figure(figsize=(24, 6))
    ecg_data = record[0][:, 1]
    plt.plot(ecg_data[:ecg_size])
    plt.plot(draw_peaks, ecg_data[draw_peaks], 'ro')
    plt.show()


def find_heartbeat_len(peaks):
    distances = []
    for i in range(len(peaks)):
        if i + 1 < len(peaks):
            distances.append(peaks[i + 1] - peaks[i])
    distances = np.array(distances)
    return int(distances.mean())


def separate_ecg_to_heartbeats(record):
    fs = record[1]['fs']
    unfiltered_ecg = record[0][:, 1]
    r_peaks = detect_r_peaks(record, fs)
    heartbeat_len = find_heartbeat_len(r_peaks)
    beats = []
    for i in range(len(r_peaks)):
        if i != 0:
            r_index = r_peaks[i]
            retreat = heartbeat_len // 2
            beats.append(unfiltered_ecg[r_index - retreat:r_index + retreat])
    return beats


def separate_ecg_to_heartbeats_with_annotation(record, annotation):
    fs = record[1]['fs']
    sample = annotation.__dict__['sample']
    marker = annotation.__dict__['symbol']
    unfiltered_ecg = record[0][:, 1]
    r_peaks = detect_r_peaks(record, fs)
    heartbeat_len = find_heartbeat_len(r_peaks)
    beats = []
    annotated_beats = []
    for i in range(len(r_peaks)):
        if i != 0:
            r_index = r_peaks[i]
            retreat = heartbeat_len // 2
            beats.append(unfiltered_ecg[r_index - retreat:r_index + retreat])
            for j in range(len(sample)):
                if r_index - retreat < sample[j]:
                    annotated_beats.append(marker[j])
                    break
    return {'beats': beats, 'annotated_beats': annotated_beats}


def select_beats_by_type(beats, annotated_beats, markers):
    selected_beats = []
    selected_annotation = []
    for i in range(len(beats)):
        if annotated_beats[i] in markers:
            selected_beats.append(beats[i])
            selected_annotation.append(annotated_beats[i])
    return {'beats': selected_beats, 'annotated_beats': selected_annotation}


def ecg_wave_detection(heart_beats):
    p_waves = []
    qrs_waves = []
    t_waves = []
    retreat = len(heart_beats[0]) // 2
    for j in range(len(heart_beats)):
        if j + 1 != len(heart_beats):
            p_waves.append(heart_beats[j][:retreat - 15])
            qrs_waves.append(heart_beats[j][retreat - 15:retreat + 15])
            t_waves.append(heart_beats[j][retreat + 15:])
    return {'p_waves': p_waves, 'qrs_waves': qrs_waves, 't_waves': t_waves}


def waves_clustering(waves, amount_clusters):
    waves = np.array(waves)
    kmeans = KMeans(init='k-means++', n_clusters=amount_clusters, n_init=10)
    kmeans.fit(waves)
    predict_cluster = kmeans.predict(waves)
    return predict_cluster


def generate_dict_symbol_to_cluster(amount_cluster, alphabet):
    return {i: alphabet[i] for i in range(amount_cluster)}


def transform_cluster_to_symbol(predicted_cluster, vocab):
    def get_item(x):
        return vocab[x]

    vfunc = np.vectorize(get_item)
    predicted_symbol = vfunc(predicted_cluster)
    return predicted_symbol


def transform_heartbeat_to_word(pt_symbol, qrs_symbol):
    words = []
    for i in range(len(qrs_symbol)):
        word = ''
        word += pt_symbol[i]
        word += qrs_symbol[i]
        word += pt_symbol[i + len(qrs_symbol) // 2]
        words.append(word)
    return words


def create_word2vec_model(words, min_count=0):
    word2vec = Word2Vec([words, ], min_count=min_count)
    return word2vec


def create_word2vec_based_on_record(record, annotation):
    data = separate_ecg_to_heartbeats_with_annotation(record, annotation)
    beats = data['beats']
    waves_data = ecg_wave_detection(beats)
    p_waves = waves_data['p_waves']
    qrs_waves = waves_data['qrs_waves']
    t_waves = waves_data['t_waves']
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    amount_cluster_pt = 15
    pt = np.array(p_waves + t_waves)
    pt_clustering = waves_clustering(pt, amount_cluster_pt)
    vocab = generate_dict_symbol_to_cluster(amount_cluster_pt,
                                            alphabet[:amount_cluster_pt])
    pt_symbols = transform_cluster_to_symbol(pt_clustering, vocab)
    amount_cluster_qrs = 10
    qrs_clustering = waves_clustering(qrs_waves, amount_cluster_qrs)
    vocab = generate_dict_symbol_to_cluster(amount_cluster_qrs,
                                            alphabet[amount_cluster_pt:])
    qrs_symbols = transform_cluster_to_symbol(qrs_clustering, vocab)
    words = transform_heartbeat_to_word(pt_symbols, qrs_symbols)
    word2vec = Word2Vec([words, ], min_count=0)
    return {'model': word2vec, 'words': words, 'beats': data['beats'],
            'annotation': data['annotated_beats']}
