#include <stdio.h>
#include <stdlib.h>
/*
 * c is the function to be learnt 
 * the hypothesis functions are : x, x^2, 2x
 */
#define NUM_HYPOTHESIS 3
/*
 * the test data have to passed as arguments in the format
 * xi c(xi) 
 */
#define NUM_TEST_DATA  10

int total_num_consistencies = 0;

float calculate_class_likelihood(int xi[], int c_xi[], int function){

	float num_failures = 0;

	int i;

	switch(function){
		case 1:
			num_failures=0;
			for(i=0; i<NUM_TEST_DATA ;i++){
				if(c_xi[i] != xi[i])
					num_failures++;
			}
			/*
			if(num_failures > 0)
				return 0;
			else
				return 1;
				*/
			return (1-(num_failures/NUM_TEST_DATA));
			break;
		case 2:
			num_failures=0;
			for(i=0; i<NUM_TEST_DATA ;i++){
				if(c_xi[i] != xi[i]*xi[i])
					num_failures++;
			}
			/*
			if(num_failures > 0)
				return 0;
			else
				return 1;
				*/
			return (1-(num_failures/NUM_TEST_DATA));
			break;
		case 3:
			num_failures=0;
			for(i=0; i<NUM_TEST_DATA ;i++){
				if(c_xi[i] != 2*xi[i])
					num_failures++;
			}
			/*
			if(num_failures > 0)
				return 0;
			else
				return 1;
				*/
			return (1-(num_failures/NUM_TEST_DATA));
			break;
	}

	fprintf(stderr, "Program error\n");
	exit(0);
}
float calculate_posterior ( int xi[], int c_xi[], int function){
	
	/*
	 * based on the formula: 
	 * P(h|D) = P(D|h)*P(h)/P(D)
	 * where P(h) = 1/NUM_HYPOTHESIS
	 * P(h|D) is posterior probability (the one we are interested in)
	 * P(D|h) is the class likelihood
	 * P(D) = total_num_consistencies/NUM_HYPOTHESIS
	 * hence when simplified the NUM_HYPOTHESIS in numerator and denominator are 
	 * eliminated 
	 * hence P(h|D) = P(D|h)/total_num_consistencies)
	 */
	return (calculate_class_likelihood(xi, c_xi, function)/(total_num_consistencies));
}

void calculate_evidence(int xi[], int c_xi[]){

	int i;
	for(i=0; i<NUM_HYPOTHESIS; i++){
		if(calculate_class_likelihood(xi, c_xi, i+1) >= 0.5 )
			total_num_consistencies++;
	}
}


int main(int argc, char *argv[]){

	int i, xi[NUM_TEST_DATA], c_xi[NUM_TEST_DATA];
	float posterior[NUM_HYPOTHESIS];


	if(argc < 2*NUM_TEST_DATA+1){
		fprintf(stderr, "Insufficient Arguments\n");
		return -1;
	}


	for(i=1;i<(NUM_TEST_DATA+1);i++){
		xi[i-1] = atoi(argv[2*i-1]);
		c_xi[i-1] = atoi(argv[2*i]);
	}

	calculate_evidence(xi, c_xi);

	printf("The total number of consistencies are : %d\n", total_num_consistencies);

	if(total_num_consistencies <= 0){
		fprintf(stderr, "There are no consistent hypothesis, cannot regress\n");
		return -1;
	}
	for(i=1;i<(NUM_TEST_DATA+1);i++){
		printf("The values are %d : %d \n",xi[i-1],c_xi[i-1]);
	}

	for(i=0;i<NUM_HYPOTHESIS;i++)
		posterior[i] = calculate_posterior(xi, c_xi, i+1);
	
		printf("The probabilities for x, x^2, 2x are :"); 
	for(i=0;i<NUM_HYPOTHESIS;i++)
		printf("[%4.3f]", posterior[i]) ;
	
	return 0;

}
