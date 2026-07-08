/**************************************************************************//**
 * @file     GestureModel.cpp
 * @brief    Gesture model op-resolver definition (mirrors YOLOv8nODModel).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "GestureModel.hpp"
#include "log_macros.h"

const tflite::MicroOpResolver &arm::app::GestureModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::GestureModel::EnlistOperations()
{
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
