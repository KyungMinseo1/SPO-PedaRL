import torch
from vllm import LLM
from vllm.config import PoolerConfig
import gc

def test_high_score_reward():
    """Test script to verify high reward scores using pedagogical examples."""
    
    reward_model_path = "eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel"
    
    print("=" * 80)
    print(f"Loading Model: {reward_model_path}")
    print("=" * 80)
    
    try:
        # 1. 모델 로드 (가장 안정적인 설정)
        reward_model = LLM(
            model=reward_model_path,
            gpu_memory_utilization=0.6, # 메모리 여유 있게
            max_model_len=2048,
            convert="classify",
            dtype=torch.float32,
            pooler_config=PoolerConfig(
                pooling_type="LAST",
                normalize=False,
                use_activation=False # Raw Score(Logit)를 얻기 위함
            ),
        )
        tokenizer = reward_model.get_tokenizer()
        print("✅ Model loaded successfully!")
        
        # 2. 테스트 프롬프트 준비
        # Case A: 아까의 그 '괜찮은' 답변 (~0.99점 예상)
        conv_standard = [
            {"role": "system", "content": "Judge the pedagogical quality of the responses."},
            {"role": "user", "content": "Problem: What is 2+2?\nReference Solution: "},
            {"role": "assistant", "content": "Let me help you think through this. What do you get when you add 2 and 2?"}
        ]

        # Case B: '아주 훌륭한' 답변 (논문 Figure 5 + Thinking Tag 활용)
        # 논문에 따르면 <think> 태그와 구체적인 소크라테스식 유도는 점수가 높습니다.
        conv_excellent = [
            {"role": "system", "content": "Judge the pedagogical quality of the responses."},
            {"role": "user", "content": "Problem: A student uses a calculator to find an answer but instead of pressing the x^2 key, presses the square root key by mistake. The student's answer was 9. What should the answer have been?\nReference Solution: 81"},
            {"role": "assistant", "content": "<think>The student made a mistake by taking the square root instead of squaring. The result is 9. To guide them, I should first ask them to reverse the operation to find the original number, rather than telling them the number directly.</think>\nThat's an interesting mistake! Let's analyze it step-by-step. If pressing the square root key gave them 9, what does that tell you about the number they started with?"}
        ]
        
        # 프롬프트 변환
        prompts = [
            tokenizer.apply_chat_template(conv_standard, tokenize=False),
            tokenizer.apply_chat_template(conv_excellent, tokenize=False)
        ]
        
        print(f"\n[Running Inference] Comparing {len(prompts)} prompts...")
        
        # 3. 인퍼런스 (.encode + classify task 사용)
        # 이 방식이 Raw Logit을 가져오는 가장 확실한 방법입니다.
        outputs = reward_model.encode(prompts, pooling_task="classify")
        
        # 점수 추출
        scores = []
        for output in outputs:
            data = output.outputs.data
            # 데이터가 리스트인지 텐서인지 스칼라인지 확인 후 처리
            if hasattr(data, 'item'):
                score = data.item()
            elif isinstance(data, (list, tuple)):
                score = data[0]
            else:
                score = data
            scores.append(score)

        # 결과 출력
        print("\n" + "="*40)
        print(" RESULT COMPARISON ")
        print("="*40)
        print(f"1. Standard Prompt Score : {scores[0]:.4f}")
        print(f"2. Excellent Prompt Score: {scores[1]:.4f}")
        
        if scores[1] > 1.0:
            print("\n✅ SUCCESS! Score > 1.0 achieved.")
            print("The model successfully recognized the higher pedagogical quality.")
        else:
            print("\n⚠️ Note: Score is still low. Check prompt format or model specifics.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 메모리 정리
        try:
            del reward_model
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        print("\n✅ Cleanup complete")

if __name__ == "__main__":
    test_high_score_reward()