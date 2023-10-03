export MODEL=model

init:
	/home/casu/init_openvino.sh

compile_model:
	cp ./ipynb/${MODEL}/tf_${MODEL}_MNIST.pbtxt ./mnist_calc_test
	cd mnist_calc_test && mo_tf.py --input_model tf_${MODEL}_MNIST.pbtxt --extensions ./user_mo_extensions \
	--output_dir . --input_shape [1,784] --data_type FP16 --input_model_is_text

clean:
	rm -f mnist_calc_test/tf_${MODEL}_*.*
