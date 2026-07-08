/**************************************************************************//**
 * @file     GestureModel.hpp
 * @brief    Gesture model (Day-1 = reused YOLOv8n-od.tflite, already Vela'd).
 *
 * Same architecture/op set as the old object-detection model: the proven
 * Transpose + EthosU resolver. Kept as its own class so it can diverge from
 * Face later without touching the test harness.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef _GESTURE_MODEL_HPP_
#define _GESTURE_MODEL_HPP_

#include "Model.hpp"

#define GESTURE_INPUT_TENSOR   (192)   /* matches the reused YOLOv8n-od model */

namespace arm
{
namespace app
{

class GestureModel : public Model
{
protected:
    const tflite::MicroOpResolver &GetOpResolver() override;
    bool EnlistOperations() override;

private:
    /* Transpose + EthosU, identical to the proven YOLOv8nODModel. */
    static constexpr int ms_maxOpCnt = 2;
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
};

} /* namespace app */
} /* namespace arm */

#endif /* _GESTURE_MODEL_HPP_ */
