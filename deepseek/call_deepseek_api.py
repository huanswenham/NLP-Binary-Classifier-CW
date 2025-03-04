from deepseek_api import DeepSeekApi

def main():
  deepseek = DeepSeekApi()
  result = deepseek.pcl_classify()
  print(result)

if __name__ == "__main__":
    main()