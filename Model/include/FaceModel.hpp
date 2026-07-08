/**************************************************************************//**
 * @file     FaceModel.hpp
 * @brief    Face-detection model (YOLOv8n-face, Vela-compiled for Ethos-U55).
 *
 * Thin subclass of the eval-kit Model base class; only the op resolver differs.
 * Vela report shows CPU operators = 0, so AddEthosU() is the essential op;
 * AddTranspose() is kept as a harmless safety net (see EnlistOperations()).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef _FACE_MODEL_HPP_
#define _FACE_MODEL_HPP_

#include "Model.hpp"

#define FACE_INPUT_TENSOR   (192)   /* imgsz used for the INT8 export */

namespace arm
{
namespace app
{

class FaceModel : public Model
{
protected:
    /** @brief  Reference to the op resolver. */
    const tflite::MicroOpResolver &GetOpResolver() override;

    /** @brief  Register the operators this model uses. */
    bool EnlistOperations() override;

private:
    /* Max distinct ops the resolver can hold. 4 leaves headroom in case a
     * future real face model keeps a couple of CPU ops after Vela. */
    static constexpr int ms_maxOpCnt = 4;
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
};

} /* namespace app */
} /* namespace arm */

#endif /* _FACE_MODEL_HPP_ */
