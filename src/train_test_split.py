def get_mapping():
	f = open("LOC_synset_mapping.txt");
	label_lines = [i for i in f.readlines()];
	f.close();

	label_mapping = dict();
	for line in label_lines:
		label_sym  = "";
		label_text = "";
		for i in range(len(line)):
			if(line[i] == " "):
				label_sym  = line[0:i];
				label_text = line[i+1:].split(", ")[0].strip();
				break;
		label_mapping[label_sym] = label_text;
	return label_mapping;

def process_lines(label_mapping):
	f = open("imagenet2015_validation_images.txt");
	lines = [i for i in f.readlines()];
	f.close();

	lbl = label_mapping[lines[0].strip().split()[0]]
	src_list = [];
	resource_dict = dict();
	for line in lines:
		label = "";
		l = line.strip().split();
		if(len(l) > 0):
			label = label_mapping[l[0]];
			src = l[1];
			if(label == lbl):
				src_list.append(src);
			else:
				resource_dict[lbl] = src_list;
				src_list = []
				lbl = label
	return resource_dict;

if __name__ == "__main__":
	mapping = get_mapping();
	resource_dict = process_lines(mapping);

	f = open("train_test/key_file_15.txt", "w");
	d = open("train_test/data_path_15.txt", "w");
	for key in resource_dict.keys():
		l = resource_dict[key];
		for i in range(15):
			if(len(l) != 0):
				f.write(key + "\n");
				d.write(l[i].split("/")[-1] + "\n");
	f.close();
	d.close();
