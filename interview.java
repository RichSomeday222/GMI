
import java.util.Arrays;

/*
 Given an integer array nums, move all 0's to the 
 end of it while maintaining the relative order of the non-zero elements. 
 Tip: Do it with O(1) space complexity.
*/
/* 1.直接修改数组
 * 2. 包含负数
 * 3. 会有很大的情况
*/
public class interview{
    public void moveZeros(int[] nums){
        int index = 0;
        for (int i = 0; i < nums.length; i++){
            if (nums[i] != 0){
                nums[index] = nums[i];
                index ++;
            }
        }
        while (index < nums.length){
            nums[index] = 0;
            index ++;
        }
    }
    public static void main(String[] args){
        interview solu = new interview();
        // Example 1
        int[] nums1 = {0, 1, 0, 2, 3, 4};
        solu.moveZeros(nums1);
        String str1 =  Arrays.toString(nums1);
        System.out.println(str1);
    }
}

/*
Take home project：
Use GMI endpoint to build a RAG (can be anything, like your resume)

https://inference-engine.gmicloud.ai/ 
API key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6Ijk5ZTMyMjI4LTk1YWMtNDdjZS1iMTc1LWI4YmJkNWZlMDYyYSIsInR5cGUiOiJpZV9tb2RlbCJ9.wf85b2eocjJvlm4PFePDfkpnjpJ0qmj2jumvAj3XGk4
 * 
*/