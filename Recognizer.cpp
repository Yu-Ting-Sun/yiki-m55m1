/**************************************************************************//**
 * @file     Recognizer.cc
 * @version  V0.10
 * @brief    Recognizer function
 * * SPDX-License-Identifier: Apache-2.0
 * @copyright (C) 2022 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "Recognizer.hpp"

#include "TensorFlowLiteMicro.hpp"
#include "log_macros.h"

#include <vector>
#include <string>
#include <set>
#include <cstdint>
#include <cinttypes>
#include <cmath>

// https://en.wikipedia.org/wiki/Cosine_similarity
static float cosine_similarity(std::vector<float> embedArray1, std::vector<float> embedArray2)
{
    int numElem;
    float ret = 0, mod1 = 0, mod2 = 0;

    if (embedArray1.size() != embedArray2.size())
        return -1;

    //embedArray1 and embedArray2 must the same size.
    numElem = embedArray1.size();

    for (int i = 0; i < numElem; i++)
    {
        ret += embedArray1[i] * embedArray2[i];
        mod1 += embedArray1[i] * embedArray1[i];
        mod2 += embedArray2[i] * embedArray2[i];
    }

    return ret / sqrt(mod1) / sqrt(mod2);
}


namespace arm
{
namespace app
{
Recognizer::Recognizer()
{
    m_threshold = 0.7;
}

bool Recognizer::GetRecognitionResults(
    TfLiteTensor *outputTensor,
    RecognitionResult &predResults,
    const std::vector <S_LABEL_INFO> &labels)
{
    predResults.m_predict = 0;

    if (outputTensor == nullptr)
    {
        printf_err("Output vector is null pointer.\n");
        return false;
    }

    uint32_t embedDimension = 1;

    for (int inputDim = 0; inputDim < outputTensor->dims->size; inputDim++)
    {
        embedDimension *= outputTensor->dims->data[inputDim];
    }

    /* De-Quantize Output Tensor */
    QuantParams quantParams = GetTensorQuantParams(outputTensor);

    /* Floating point tensor data to be populated
     * NOTE: The assumption here is that the output tensor size isn't too
     * big and therefore, there's neglibible impact on heap usage. */
    std::vector<float> embedArray(embedDimension);

    /* Populate the floating point buffer */
    switch (outputTensor->type)
    {
        case kTfLiteUInt8:
        {
            uint8_t *tensor_buffer = tflite::GetTensorData<uint8_t>(outputTensor);

            for (size_t i = 0; i < embedDimension; ++i)
            {
                embedArray[i] = quantParams.scale *
                                (static_cast<float>(tensor_buffer[i]) - quantParams.offset);
            }

            break;
        }

        case kTfLiteInt8:
        {
            int8_t *tensor_buffer = tflite::GetTensorData<int8_t>(outputTensor);

            for (size_t i = 0; i < embedDimension; ++i)
            {
                embedArray[i] = quantParams.scale *
                                (static_cast<float>(tensor_buffer[i]) - quantParams.offset);
            }

            break;
        }

        case kTfLiteFloat32:
        {
            float *tensor_buffer = tflite::GetTensorData<float>(outputTensor);

            for (size_t i = 0; i < embedDimension; ++i)
            {
                embedArray[i] = tensor_buffer[i];
            }

            break;
        }

        default:
            printf_err("Tensor type %s not supported by classifier\n",
                       TfLiteTypeGetName(outputTensor->type));
            return false;
    }

    /* Score each PERSON, not each reference: the mean of the label's best
     * RECOG_TOP_K references. A plain max over all references favoured
     * whoever had the most of them — with 24 camera shots of one person
     * against a single selfie each, the selfie owners kept being read as
     * the 24-shot person. */
    const int RECOG_TOP_K = 3;
    struct PersonScore
    {
        const std::string *label;
        float top[RECOG_TOP_K];
        int   n;
    };
    std::vector<PersonScore> people;

    for (size_t i = 0; i < labels.size(); i++)
    {
        float d = cosine_similarity(embedArray, labels[i].fParam);

        PersonScore *ps = nullptr;
        for (auto &p : people)
            if (*p.label == labels[i].szLable) { ps = &p; break; }
        if (!ps)
        {
            people.push_back({ &labels[i].szLable, { -1.0f, -1.0f, -1.0f }, 0 });
            ps = &people.back();
        }

        /* keep top[] sorted best-first */
        if (ps->n < RECOG_TOP_K) ps->n++;
        for (int k = 0; k < ps->n; k++)
            if (d > ps->top[k])
            {
                for (int m = ps->n - 1; m > k; m--) ps->top[m] = ps->top[m - 1];
                ps->top[k] = d;
                break;
            }
    }

    int    bestIdx = -1, secondIdx = -1;
    double best = -1.0, second = -1.0;
    for (size_t i = 0; i < people.size(); i++)
    {
        double s = 0;
        for (int k = 0; k < people[i].n; k++) s += people[i].top[k];
        s /= people[i].n;

        if (s > best)
        {
            second = best; secondIdx = bestIdx;
            best = s;      bestIdx = (int)i;
        }
        else if (s > second)
        {
            second = s; secondIdx = (int)i;
        }
    }

    predResults.m_predict = best;
    predResults.m_second  = second;
    predResults.m_secondLabel = (secondIdx >= 0) ? *people[secondIdx].label : "";
    predResults.m_recognize = false;

    if (bestIdx >= 0 && best > m_threshold &&
        (secondIdx < 0 || best - second >= m_margin))
    {
        predResults.m_label = *people[bestIdx].label;
        predResults.m_recognize = true;
        return true;
    }

    if (bestIdx >= 0)
        predResults.m_label = *people[bestIdx].label;   /* for the log only */
    return false;
}

void Recognizer::SetThreshold(double threshold)
{
    m_threshold = threshold;
}

void Recognizer::SetMargin(double margin)
{
    m_margin = margin;
}


} /* namespace app */
} /* namespace arm */
