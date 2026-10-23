#include "ninfer/ops/host_vision.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace ninfer;
namespace fixture = ninfer::test::quantized_weight;

void require(bool ok, const char* message) {
    if (!ok) { throw std::runtime_error(message); }
}

// Independent nearest-even BF16 oracle using distances to adjacent represented values.
double rounded(double value) {
    const float source = static_cast<float>(value);
    const auto word = std::bit_cast<std::uint32_t>(source);
    const auto base = word & 0xffff0000U;
    const float first = std::bit_cast<float>(base);
    const float second = std::bit_cast<float>(base + 0x10000U);
    const double a = std::abs(value - first), b = std::abs(value - second);
    return a < b || (a == b && ((base >> 16) & 1) == 0) ? first : second;
}

void close(float actual, double expected, const char* label) {
    const double target = rounded(expected);
    if (!std::isfinite(actual) || std::abs(actual - target) > 0.008 * std::abs(target) + 0.00003) {
        std::cerr << label << " actual=" << actual << " reference=" << target << '\n';
        throw std::runtime_error(label);
    }
}

void linear() {
    struct Shape { QType type; int n, k; };
    for (const auto shape : {Shape{QType::Q6G64_F16S,1152,1536}, {QType::Q4G64_F16S,3456,1152},
            {QType::Q5G64_F16S,1152,1152}, {QType::Q4G64_F16S,4304,1152},
            {QType::Q5G64_F16S,1152,4304}, {QType::W8G32_F16S,4608,4608},
            {QType::W8G32_F16S,5120,4608}, {QType::W8G32_F16S,2048,4608}}) {
        auto packed = fixture::make_patterned_weight(shape.type, shape.n, shape.k, 33,
            {.row_split_scale=fixture::RowSplitScalePattern::Small,
             .row_split_codes=fixture::RowSplitCodePattern::Hashed});
        const auto weight = ops::decode_host_linear(packed.device_weight(packed.payload.data()));
        // Check every decoded weight against the fixture's independently addressed FP64 codec.
        for (int row = 0; row < shape.n; ++row) {
            for (int k = 0; k < shape.k; ++k) {
                require(weight.values[std::size_t(row)*shape.k+k] == fixture::logical_weight_fp64(packed,row,k),
                        "CPU weight decode differs from exact signed-code/scale oracle");
            }
        }
        std::vector<float> input(std::size_t(shape.k)*3);
        for (std::size_t i=0;i<input.size();++i) { input[i]=static_cast<float>(rounded(std::sin(i*0.071)*0.25)); }
        const auto output=ops::host_linear(input,weight);
        for(int token=0;token<3;++token) {
            for(const int row : {0,shape.n/2,shape.n-1}) {
                double total=0;
                for(int k=0;k<shape.k;++k) { total+=double(input[std::size_t(token)*shape.k+k])*fixture::logical_weight_fp64(packed,row,k); }
                close(output[std::size_t(token)*shape.n+row],total,"CPU linear FP64 oracle");
            }
        }
        std::cout<<"linear N="<<shape.n<<" K="<<shape.k<<" PASS\n";
    }
}

void attention() {
    constexpr int heads=16,dim=72,hidden=heads*dim,tokens=270;
    const std::vector<std::int32_t> bounds{0,3,134,270};
    std::vector<float> qkv(std::size_t(tokens)*3*hidden);
    for(std::size_t i=0;i<qkv.size();++i) { qkv[i]=static_cast<float>(rounded(std::sin(i*0.019)*0.5)); }
    const auto result=ops::host_vision_attention(qkv,bounds,heads,dim);
    for(std::size_t segment=1;segment<bounds.size();++segment) {
        for(const int query : {bounds[segment-1],(bounds[segment-1]+bounds[segment])/2,bounds[segment]-1}) {
            for(int head=0;head<heads;++head) {
                std::vector<double> scores(bounds[segment]-bounds[segment-1]);
                for(int key=bounds[segment-1];key<bounds[segment];++key) {
                    double dot=0;
                    for(int d=0;d<dim;++d) {
                        dot+=double(qkv[std::size_t(query)*3*hidden+head*dim+d])*
                                   qkv[std::size_t(key)*3*hidden+hidden+head*dim+d];
                    }
                    scores[key-bounds[segment-1]]=dot/std::sqrt(double(dim));
                }
                const double maximum=*std::max_element(scores.begin(),scores.end());
                double sum=0;
                for(auto& value:scores) {value=std::exp(value-maximum);sum+=value;}
                for(int d=0;d<dim;++d) {
                    double value=0;
                    for(int key=bounds[segment-1];key<bounds[segment];++key) {
                        value+=scores[key-bounds[segment-1]]/sum*qkv[std::size_t(key)*3*hidden+2*hidden+head*dim+d];
                    }
                    close(result[std::size_t(query)*hidden+head*dim+d],value,"CPU segmented attention FP64 oracle");
                }
            }
        }
    }
    std::fill(qkv.begin()+std::size_t(3)*3*hidden,qkv.end(),17.0F);
    const auto changed=ops::host_vision_attention(qkv,bounds,heads,dim);
    require(std::equal(result.begin(),result.begin()+3*hidden,changed.begin()),"CPU attention leaks across video segments");
    std::cout<<"attention temporal segments, block boundaries and full-key softmax PASS\n";
}

void elementwise() {
    constexpr int hidden=1152,tokens=3;
    std::vector<float> input(hidden*tokens),weight(hidden),bias(hidden);
    for(int i=0;i<hidden;++i) {weight[i]=float(rounded(0.5+std::sin(i)*0.3));bias[i]=float(rounded(std::cos(i)*0.1));}
    for(int i=0;i<hidden*tokens;++i) {input[i]=float(rounded((i/hidden==1?1000:0)+std::sin(i*0.1)));}
    const auto output=ops::host_layer_norm(input,weight,bias,1e-6F);
    for(int t=0;t<tokens;++t) {
        double mean=0,variance=0;
        for(int j=0;j<hidden;++j) {mean+=input[t*hidden+j];} mean/=hidden;
        for(int j=0;j<hidden;++j) {variance+=std::pow(input[t*hidden+j]-mean,2);} variance/=hidden;
        for(int j=0;j<hidden;++j) {close(output[t*hidden+j],(input[t*hidden+j]-mean)/std::sqrt(variance+1e-6)*weight[j]+bias[j],"CPU LayerNorm FP64 oracle");}
    }
    for(bool exact:{false,true}) {
        std::vector<float> values(65);
        for(int i=0;i<65;++i) {values[i]=float(rounded((i-32)*0.2));}
        const auto source=values; ops::host_gelu(values,exact);
        for(int i=0;i<65;++i) {
            const double x=source[i];
            close(values[i],0.5*x*(1+(exact?std::erf(x/std::sqrt(2.0)):
                  std::tanh(std::sqrt(2.0/std::acos(-1.0))*(x+0.044715*x*x*x)))),"CPU GELU FP64 oracle");
        }
    }
    // Exact BF16 transport and ties, including infinities/NaNs, do not rely on the Op helpers.
    for(unsigned i=0;i<65536;++i) {
        const auto value=std::bit_cast<float>(i<<16);
        const auto encoded=ops::host_float_to_bf16(value);
        if(std::isnan(value)) {require(std::isnan(ops::host_bf16_to_float(encoded)),"BF16 NaN lost");}
        else {require(encoded==i,"BF16 exact represented value changed");}
    }
    std::cout<<"LayerNorm, GELU and BF16 representation PASS\n";
}

void positions() {
    constexpr int heads=16,dim=72,hidden=heads*dim,tokens=4;
    std::vector<float> qkv(tokens*3*hidden);
    for(std::size_t i=0;i<qkv.size();++i) {qkv[i]=float(rounded(std::sin(i*0.11)));}
    const auto original=qkv;
    const std::vector<std::int32_t> positions{0,3,8,15,7,4,0,2};
    ops::host_vision_rope(qkv,positions,heads,dim);
    for(int t=0;t<tokens;++t) {
        for(int head=0;head<heads;++head) {
            for(int pair=0;pair<dim/2;++pair) {
                const double angle=positions[(pair/18)*tokens+t]*std::pow(10000.0,-double(pair%18)/18);
                for(int plane=0;plane<2;++plane) {
                    const auto i=std::size_t(t)*3*hidden+plane*hidden+head*dim+pair;
                    close(qkv[i],original[i]*std::cos(angle)-original[i+36]*std::sin(angle),"CPU 2D RoPE FP64 oracle");
                    close(qkv[i+36],original[i+36]*std::cos(angle)+original[i]*std::sin(angle),"CPU 2D RoPE FP64 oracle");
                }
            }
            for(int d=0;d<dim;++d) {
                const auto i=std::size_t(t)*3*hidden+2*hidden+head*dim+d;
                require(qkv[i]==original[i],"Vision RoPE changed V");
            }
        }
    }
    std::vector<float> x(tokens*hidden), table(48*48*hidden);
    for(std::size_t i=0;i<table.size();++i) {table[i]=float(rounded(std::sin(i*0.007)));}
    const std::vector<std::int32_t> indices{0,47,48,95, 203,204,251,252, 2260,2261,2302,2303, 1,3,5,7};
    const std::vector<float> weights{0.125F,0.375F,0.125F,0.375F, 1,0,0,0, 0.5F,0,0.5F,0, 0.25F,0.25F,0.25F,0.25F};
    ops::host_vision_pos_embed(x,table,indices,weights,hidden);
    for(int t=0;t<tokens;++t) {
        for(int d=0;d<hidden;++d) {
            double value=0;
            for(int j=0;j<4;++j) {value+=double(table[indices[t*4+j]*hidden+d])*weights[t*4+j];}
            close(x[t*hidden+d],value,"CPU bilinear position FP64 oracle");
        }
    }
    std::cout<<"2D RoPE, unchanged V and bilinear position embedding PASS\n";
}
} // namespace

int main() {
    try {linear(); attention(); elementwise(); positions();}
    catch(const std::exception& error) {std::cerr<<error.what()<<'\n'; return 1;}
    return 0;
}
