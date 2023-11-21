from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    try:
        generated_text = ''
        while True:
            user_input = input("\033[92m> Enter code prefix to generate (or empty to continue) (or 'ctrl+c' to exit):\n\033[0m> ")
            if user_input:
                # if input is not empty,generate new code
                while True: 
                    line = input("> ")
                    if line == '':
                        break
                    user_input += '\n'+line
                generated_text = user_input
                print('\033[92m> Generated Code:\033[0m')
            else:
                # else continue the last round code
                print('\033[92m> Continued Code:\033[0m')
            print(f'{generated_text}', end='', flush=True)
            # print('> '+'\n> '.join(generated_text.split('\n')))
            # 流式
            for _ in range(10):  # 生成 100 个字符
                input_ids = tokenizer.encode(generated_text, return_tensors="pt").cuda()
                with torch.no_grad():
                    output = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, max_length=len(input_ids[0])+10, num_return_sequences=1, no_repeat_ngram_size=2, do_sample=True, top_k=50, top_p=0.9, temperature=0.9)
                    generated_text2 = tokenizer.decode(output[0], skip_special_tokens=True)
                    predicted_token = generated_text2[len(generated_text):]
                    if not predicted_token:
                        break
                    generated_text = generated_text2
                    print(predicted_token, end='', flush=True)
                    # if predicted_token.endswith('\n'):
                    #     print('  ', end='', flush=True)
            print('\n')
    except KeyboardInterrupt:
        print("\n\033[94m> Goodbye.\033[0m")

def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Code Generation")
    parser.add_argument(
        "--model_name_or_path", "-m", type=str, help="model_name_or_path",
    )
    args = parser.parse_args()
    main(args.model_name_or_path)
    
if __name__ == "__main__":
    cli_main()