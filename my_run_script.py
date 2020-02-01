import gpt_2_simple as gpt2
import os
import requests
import click

@click.command()
@click.option(
    "-m",
    "--model",
    type=str,
    required=True,
    help="name of the gpt2 model to use from [ 124M, 355M, 774M, 1558M ]"
)
@click.option(
    "--steps",
    type=int,
    required=True,
    help="number of steps to run training for"
)
@click.option(
    "--batch-size",
    type=int,
		default=1,
    required=False,
    help="size of the batch of data to train with"
)
def main(
	model,
	steps,
	batch_size
):
	model_name = model
	if not os.path.isdir(os.path.join("models", model_name)):
		print("Downloading {} model...".format(model_name))
		gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


	file_name = "shakespeare.txt"
	if not os.path.isfile(file_name):
		url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
		data = requests.get(url)
		
		with open(file_name, 'w') as f:
			f.write(data.text)
			

	sess = gpt2.start_tf_sess()
	gpt2.finetune(sess,
								file_name,
								model_name=model_name,
								steps=steps, # steps is max number of training steps
								batch_size=batch_size,
								accumulate_gradients=1
	)   

	# gpt2.generate(sess)

if __name__ == "__main__":
		main()