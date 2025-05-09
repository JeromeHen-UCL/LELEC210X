#include "pca.h"

#if defined(ENABLE_EMB_PRECOMP) && (ENABLE_EMB_PRECOMP == 1)

void pca_transform(const float* input, float* output)
{
    for (int i = 0; i < N_COMPONENTS; i++)
    {
        output[i] = 0.0f;
        for (int j = 0; j < N_FEATURES; j++)
        {
            output[i] += PCA_COMPONENTS[i][j] * input[j];
        }
    }
}

#endif // ENABLE_EMB_PRECOMP) && (ENABLE_EMB_PRECOMP == 1)
