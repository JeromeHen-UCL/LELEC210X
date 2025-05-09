#include "scaler.h"

#if defined(ENABLE_EMB_PRECOMP) && (ENABLE_EMB_PRECOMP == 1)

void scaler_transform(const float* input, float* output)
{
    for (int i = 0; i < N_FEATURES; i++)
    {
        output[i] = (input[i] - SCALER_MEAN[i]) / SCALER_STD[i];
    }
}

#endif // ENABLE_EMB_PRECOMP) && (ENABLE_EMB_PRECOMP == 1)
