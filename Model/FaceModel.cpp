/**************************************************************************//**
 * @file     FaceModel.cpp
 * @brief    Face model op-resolver definition.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "FaceModel.hpp"
#include "log_macros.h"

const tflite::MicroOpResolver &arm::app::FaceModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::FaceModel::EnlistOperations()
{
    /* Vela report: CPU operators = 0 -> everything runs on the NPU, so the
     * Ethos-U custom op is all that's strictly required. Transpose is kept as a
     * safety net (registering an unused op is harmless; a MISSING op would make
     * AllocateTensors fail). */
    this->m_opResolver.AddTranspose();

#if defined(ARM_NPU)

    if (kTfLiteOk == this->m_opResolver.AddEthosU())
    {
        info("Added %s support to op resolver\n",
             tflite::GetString_ETHOSU());
    }
    else
    {
        printf_err("Failed to add Arm NPU support to op resolver.\n");
        return false;
    }

#endif /* ARM_NPU */
    return true;
}
