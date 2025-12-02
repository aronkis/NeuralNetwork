#!/usr/bin/env python3
import zmq
import numpy as np
import argparse
import os
import sys
from datetime import datetime

# Modulation labels
MODULATIONS = {
	0: 'BPSK',
	1: 'QPSK', 
	2: '16QAM',
	3: '32QAM'
}

FLOATS_PER_FRAME = 4097
BYTES_PER_FRAME = FLOATS_PER_FRAME * 4
IQ_FLOATS = 4096

def normalize_sample(iq_data):
	max_abs = np.max(np.abs(iq_data))
	if max_abs > 1e-10:
		return iq_data / max_abs
	return iq_data


def capture_labeled_data(zmq_address, output_dir, samples_per_class, train_ratio=0.8):
   
	total_samples = samples_per_class * 4
	
	print("=" * 60)
	print("  PLUTO SDR LABELED DATA CAPTURE")
	print("=" * 60)
	print(f"\n📡 ZeroMQ Address: {zmq_address}")
	print(f"📁 Output Directory: {output_dir}")
	print(f"📊 Samples per class: {samples_per_class}")
	print(f"📊 Total samples: {total_samples} ({samples_per_class} x 4 modulations)")
	print(f"📈 Train/Test ratio: {train_ratio:.0%}/{1-train_ratio:.0%}")
	print(f"📦 Frame size: {BYTES_PER_FRAME} bytes ({FLOATS_PER_FRAME} floats)")
	
	os.makedirs(output_dir, exist_ok=True)
	
	context = zmq.Context()
	socket = context.socket(zmq.SUB)
	socket.connect(zmq_address)
	socket.setsockopt(zmq.SUBSCRIBE, b'')
	socket.setsockopt(zmq.RCVTIMEO, 10000) 
	
	print(f"\n✅ Connected to ZeroMQ at {zmq_address}")
	print("\n⏳ Waiting for data from GNU Radio...")
	
	print(f"Target: {samples_per_class} samples per class\n")
	
	all_data = []
	all_labels = []
	class_counts = {i: 0 for i in range(4)}
	
	def all_classes_complete():
		return all(class_counts[i] >= samples_per_class for i in range(4))
	
	try:
		while not all_classes_complete():
			try:
				raw_data = socket.recv()
				
				if len(raw_data) % BYTES_PER_FRAME != 0:
					print(f"Message size {len(raw_data)} not a multiple of frame size {BYTES_PER_FRAME}")
					continue
				
				floats = np.frombuffer(raw_data, dtype=np.float32)
				num_frames = len(floats) // FLOATS_PER_FRAME
				
				for frame_idx in range(num_frames):
					if all_classes_complete():
						break
					
					start = frame_idx * FLOATS_PER_FRAME
					frame_floats = floats[start:start + FLOATS_PER_FRAME]
					
					iq_data = frame_floats[:IQ_FLOATS].copy()
					label = int(frame_floats[IQ_FLOATS])
					
					if label not in MODULATIONS:
						print(f"Invalid label: {label}, skipping frame")
						continue
					
					if class_counts[label] >= samples_per_class:
						continue
					
					iq_normalized = normalize_sample(iq_data)
					
					all_data.append(iq_normalized)
					all_labels.append(label)
					class_counts[label] += 1
				
				total_captured = len(all_data)
				if total_captured % 100 == 0 and total_captured > 0:
					status = []
					for i in range(4):
						if class_counts[i] >= samples_per_class:
							status.append(f"✓{MODULATIONS[i]}:{class_counts[i]}")
						else:
							remaining = samples_per_class - class_counts[i]
							status.append(f"{MODULATIONS[i]}:{class_counts[i]} (need {remaining})")
					
					complete = sum(1 for i in range(4) if class_counts[i] >= samples_per_class)
					print(f"\n   [{complete}/4 classes complete] Total: {total_captured}")
					print(f"   {' | '.join(status)}")
					
					needed = [f"{MODULATIONS[i]}" for i in range(4) if class_counts[i] < samples_per_class]
					if needed:
						print(f"SWITCH TO: {', '.join(needed)} in GNU Radio GUI!")
					
			except zmq.error.Again:
				print("⚠️ Timeout waiting for data. Is GNU Radio running?")
				total_captured = sum(class_counts.values())
				if total_captured > 0:
					print(f"   Have {total_captured} samples. Continue waiting or Ctrl+C to save.")
					needed = [MODULATIONS[i] for i in range(4) if class_counts[i] < samples_per_class]
					if needed:
						print(f"Still need: {', '.join(needed)}")
				continue
				
	except KeyboardInterrupt:
		print(f"\n\n⚠️ Interrupted by user after {sum(class_counts.values())} samples")
	
	finally:
		socket.close()
		context.term()
	
	if len(all_data) == 0:
		print("\nNo data captured. Exiting.")
		return
	
	data_array = np.array(all_data, dtype=np.float32)
	labels_array = np.array(all_labels, dtype=np.int32)
	
	min_count = min(np.sum(labels_array == i) for i in range(4))
	print(f"\nBalancing dataset: {min_count} samples per class")
	
	balanced_data = []
	balanced_labels = []
	for label in range(4):
		mask = labels_array == label
		class_data = data_array[mask][:min_count]
		class_labels = labels_array[mask][:min_count]
		balanced_data.append(class_data)
		balanced_labels.append(class_labels)
	
	data_array = np.vstack(balanced_data)
	labels_array = np.hstack(balanced_labels)
	
	indices = np.random.permutation(len(data_array))
	data_array = data_array[indices]
	labels_array = labels_array[indices]
	
	train_data_list = []
	train_labels_list = []
	test_data_list = []
	test_labels_list = []
	
	for label in range(4):
		mask = labels_array == label
		class_data = data_array[mask]
		class_labels = labels_array[mask]
		split_idx = int(len(class_data) * train_ratio)
		train_data_list.append(class_data[:split_idx])
		train_labels_list.append(class_labels[:split_idx])
		test_data_list.append(class_data[split_idx:])
		test_labels_list.append(class_labels[split_idx:])
	
	train_data = np.vstack(train_data_list)
	train_labels = np.hstack(train_labels_list)
	test_data = np.vstack(test_data_list)
	test_labels = np.hstack(test_labels_list)
	
	train_indices = np.random.permutation(len(train_data))
	train_data = train_data[train_indices]
	train_labels = train_labels[train_indices]
	
	test_indices = np.random.permutation(len(test_data))
	test_data = test_data[test_indices]
	test_labels = test_labels[test_indices]
	
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	
	train_data_file = os.path.join(output_dir, "pluto_train_data.csv")
	train_labels_file = os.path.join(output_dir, "pluto_train_labels.csv")
	test_data_file = os.path.join(output_dir, "pluto_test_data.csv")
	test_labels_file = os.path.join(output_dir, "pluto_test_labels.csv")
	
	np.savetxt(train_data_file, train_data, delimiter=',', fmt='%.6f')
	np.savetxt(train_labels_file, train_labels, delimiter=',', fmt='%d')
	np.savetxt(test_data_file, test_data, delimiter=',', fmt='%.6f')
	np.savetxt(test_labels_file, test_labels, delimiter=',', fmt='%d')
	
	print("\n" + "=" * 60)
	print("  CAPTURE COMPLETE")
	print("=" * 60)
	print(f"\nTotal samples captured: {len(data_array)}")
	print(f"Data shape: {data_array.shape}")
	print(f"Training samples: {len(train_data)}")
	print(f"Test samples: {len(test_data)}")
	
	print("\nSamples per class:")
	for label, name in MODULATIONS.items():
		count = np.sum(labels_array == label)
		train_count = np.sum(train_labels == label)
		test_count = np.sum(test_labels == label)
		print(f"   {name} (label {label}): {count} total ({train_count} train, {test_count} test)")
	
	print(f"\nFiles saved:")
	print(f"   {train_data_file}")
	print(f"    {train_labels_file}")
	print(f"   {test_data_file}")
	print(f"    {test_labels_file}")
	
	info_file = os.path.join(output_dir, "pluto_info.txt")
	with open(info_file, 'w') as f:
		f.write("PLUTO SDR REAL DATA CAPTURE\n")
		f.write("=" * 50 + "\n\n")
		f.write(f"Capture timestamp: {timestamp}\n")
		f.write(f"ZeroMQ address: {zmq_address}\n")
		f.write(f"Total samples: {len(data_array)}\n")
		f.write(f"Training samples: {len(train_data)}\n")
		f.write(f"Test samples: {len(test_data)}\n")
		f.write(f"Data shape: {data_array.shape}\n\n")
		f.write("Samples per class:\n")
		for label, name in MODULATIONS.items():
			count = np.sum(labels_array == label)
			f.write(f"  {name} (label {label}): {count}\n")
		f.write(f"\nData source:\n")
		f.write(f"  Raw I/Q from blocks_multiply_xx_0_0 (after freq shift, before AGC)\n")
		f.write(f"  This matches what gnu_eval.cc receives for inference\n")
		f.write(f"\nData format:\n")
		f.write(f"  {IQ_FLOATS} floats per sample (2048 I/Q pairs interleaved)\n")
		f.write(f"  Format: [I0, Q0, I1, Q1, ..., I2047, Q2047]\n")
		f.write(f"  Per-sample normalization to [-1, +1]\n")
	
	print(f"   {info_file}")
	print("\nData ready for training!")
	print(f"\n To train with this data, update radio_train.cc to use:")
	print(f"   {train_data_file}")
	print(f"   {train_labels_file}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Capture labeled Pluto SDR data for training')
	parser.add_argument('--zmq', type=str, default='tcp://127.0.0.1:5555',
						help='ZeroMQ address (default: tcp://127.0.0.1:5555)')
	parser.add_argument('--output', type=str, default='../data/Pluto',
						help='Output directory (default: ../data/Pluto)')
	parser.add_argument('--samples', type=int, default=1000,
						help='Samples PER CLASS to capture (default: 1000, total will be 4000)')
	parser.add_argument('--train-ratio', type=float, default=0.8,
						help='Training data ratio (default: 0.8)')
	
	args = parser.parse_args()
	
	capture_labeled_data(args.zmq, args.output, args.samples, args.train_ratio)
