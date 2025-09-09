import os
from datasets import load_from_disk

ip_filter_func = lambda secret: (secret['value'][:4] == '127.' or secret['value'][:3] == '10.' or secret['value'][:8] == '192.168.' 
                                 or secret['value'][:8] == '169.254.' or (secret['value'][:4] == '172.' and eval(secret['value'][4:6]) >= 16 and eval(secret['value'][4:6]) <= 31))
email_filter_func = lambda secret: ('example' in secret['value'] or 'test' in secret['value'] or 
                                    'user' in secret['value'] or 'aaa' in secret['value'] or 'bbb' in secret['value'] or 'ccc' in secret['value'])
key_filter_func = lambda secret: (secret['value'] == 'ghp' or 'aaaaaaa' in secret['value'] or 'AAAAAAA' in secret['value'] or
                    'xxxxxxx' in secret['value'] or 'XXXXXXX' in secret['value'] or 'https' in secret['value'] or 'dummy' in secret['value'] or 
                    'placeholder' in secret['value'] or 'changeme' in secret['value'])

secret_filter_func = lambda secret: ((secret['tag'] == 'IP_ADDRESS' and ip_filter_func(secret)) or
                (secret['tag'] == 'EMAIL' and email_filter_func(secret)) or
                (secret['tag'] == 'KEY' and key_filter_func(secret)))


def filter_secrets(example):
    if type(example['secrets']) != str:
        return False
    secrets = eval(example['secrets'])
    if example['number_secrets'] == 1:
        secret = secrets[0]
        # Check if the only secret is either a local IP or an email containing "example"
        if secret_filter_func(secret):
        # if secret['start'] < 512 or secret_filter_func(secret):
            return False  # This will remove the example
    elif example['number_secrets'] > 1:
        # Filter out specific secrets
        filtered_secrets = [secret for secret in secrets
                            if not secret_filter_func(secret)]
        # filtered_secrets = [secret for secret in secrets
        #                     if not (secret['start'] < 512 or secret_filter_func(secret))]
        if len(filtered_secrets) == 0:
            return False  # This will remove the example
    return True


def update_example(example):
    secrets = eval(example['secrets'])
    # Filter out specific secrets
    filtered_secrets = [secret for secret in secrets
                        if not secret_filter_func(secret)]
    # filtered_secrets = [secret for secret in secrets
    #                     if not ((secret['start'] < 512 or secret_filter_func(secret)))]
    example['secrets'] = str(filtered_secrets)
    example['number_secrets'] = len(filtered_secrets)
    return example


def main():    
    dataset_path = './codeparrot-clean-train-secrets-filtered'
    if os.path.exists(dataset_path):
        ds_pii = load_from_disk(dataset_path)
    else:
        ds_pii = load_from_disk('codeparrot-clean-train-secrets')
        ds_pii = ds_pii.filter(filter_secrets, num_proc=48)
        ds_pii = ds_pii.map(update_example, num_proc=48)
        ds_pii.save_to_disk(dataset_path)
    print(ds_pii)


if __name__ == '__main__':
    main()
