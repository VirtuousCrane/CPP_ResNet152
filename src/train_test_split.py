f = open("imagenet2015_validation_images.txt");
lines = [i for i in f.readlines()];
f.close();

lbl = "";
src_list = []
resource_dict = dict();

for line in lines:
	l = line.strip().split();
	if(len(l) > 0):
		label = l[0];
		src = l[1];
	if(label == lbl):
		src_list.append(src);
	else:
		resource_dict[lbl] = src_list;
		src_list = []
		lbl = label

f = open("train_test/key_file_15.txt", "a+");
for key in resource_dict.keys():
	for i in range(15):
		l = resource_dict[key];
		if len(l) != 0:
			f.write(key + " " + l[i] + "\n");
f.close();
