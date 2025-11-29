// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using System.IO;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

namespace GaussianSplatting.Runtime
{
    [BurstCompile]
    public class GaussianSplatImporter
    {
        public enum DataQuality
        {
            VeryHigh,
            High,
            Medium,
            Low,
            VeryLow,
            Custom,
        }

        private readonly DataQuality _quality;
        private readonly GaussianSplatAsset.VectorFormat _formatPos;
        private readonly GaussianSplatAsset.VectorFormat _formatScale;
        private readonly GaussianSplatAsset.ColorFormat _formatColor;
        private readonly GaussianSplatAsset.SHFormat _formatSH;

        public GaussianSplatImporter(
            DataQuality quality = DataQuality.Medium,
            GaussianSplatAsset.VectorFormat formatPosition = GaussianSplatAsset.VectorFormat.Float32,
            GaussianSplatAsset.VectorFormat formatScale  = GaussianSplatAsset.VectorFormat.Float32,
            GaussianSplatAsset.ColorFormat formatColor = GaussianSplatAsset.ColorFormat.Float32x4,
            GaussianSplatAsset.SHFormat formatSH = GaussianSplatAsset.SHFormat.Float32)
        {
            _quality  = quality;
            _formatPos = formatPosition;
            _formatScale = formatScale;
            _formatColor = formatColor;
            _formatSH = formatSH;
        }

        public GaussianSplatAsset Load(string filePath)
        {
            GaussianSplatAsset splatAsset = null;

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException(filePath);
            }

            GaussianFileReader.ReadFile(filePath, out var splatData);
            splatAsset = CreateSplatAsset(splatData);
            splatData.Dispose();

            splatAsset.name = Path.GetFileNameWithoutExtension(filePath);
            return splatAsset;
        }

        public GaussianSplatAsset Load(byte[] data, string name)
        {
            GaussianSplatAsset splatAsset = null;

            if (data == null || data.Length == 0)
            {
                throw new ArgumentException("Input data is null or empty", nameof(data));
            }

            GaussianFileReader.ReadData(data, name, out var splatData);
            splatAsset = CreateSplatAsset(splatData);
            splatData.Dispose();

            splatAsset.name = name;
            return splatAsset;
        }

        unsafe GaussianSplatAsset CreateSplatAsset(NativeArray<InputSplatData> splatData)
        {
            if (!splatData.IsCreated || splatData.Length == 0)
            {
                throw new IOException("Input splat data is null or empty");
            }

            float3 boundsMin, boundsMax;
            var boundsJob = new CalcBoundsJob
            {
                BoundsMin = &boundsMin,
                BoundsMax = &boundsMax,
                SplatData = splatData
            };
            boundsJob.Schedule().Complete();

            ReorderMorton(splatData, boundsMin, boundsMax);

            // Cluster spherical harmonics
            NativeArray<int> splatSHIndices = default;
            NativeArray<GaussianSplatAsset.SHTableItemFloat16> clusteredSHs = default;
            if (_formatSH >= GaussianSplatAsset.SHFormat.Cluster64k)
            {
                ClusterSHs(splatData, _formatSH, out clusteredSHs, out splatSHIndices);
            }

            var splatAsset = ScriptableObject.CreateInstance<GaussianSplatAsset>();
            splatAsset.Initialize(splatData.Length, _formatPos, _formatScale, _formatColor, _formatSH, 
                                boundsMin, boundsMax, cameraInfos: null);

            var dataHash = new Hash128((uint)splatAsset.splatCount, (uint)splatAsset.formatVersion, 0, 0);

            // If we are using full lossless (FP32) data, then do not use any chunking, and keep data as-is
            var useChunks = _formatPos != GaussianSplatAsset.VectorFormat.Float32 ||
                            _formatScale != GaussianSplatAsset.VectorFormat.Float32 ||
                            _formatColor != GaussianSplatAsset.ColorFormat.Float32x4 ||
                            _formatSH != GaussianSplatAsset.SHFormat.Float32;
            var chunkData = useChunks ? CreateChunkData(splatData, ref dataHash) : null;
            CreatePositionsData(splatData, _formatPos, out var posData, ref dataHash);
            CreateColorData(splatData, _formatColor, out var colorData, ref dataHash);
            CreateSHData(splatData, clusteredSHs, _formatSH, out var shData, ref dataHash);
            CreateOtherData(splatData, splatSHIndices, _formatScale, out var otherData, ref dataHash);

            splatAsset.SetDataHash(dataHash);
            splatAsset.SetAssetFiles(chunkData, posData, otherData, colorData, shData);

            clusteredSHs.Dispose();
            splatSHIndices.Dispose();

            return splatAsset;
        }

        static TextAsset CreateTextAsset<T>(NativeArray<T> data, ref Hash128 dataHash) where T : unmanaged
        {
            dataHash.Append(data);
            return new TextAsset(data.Reinterpret<byte>(UnsafeUtility.SizeOf<T>()));
        }

        [BurstCompile]
        struct CalcBoundsJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public unsafe float3* BoundsMin;
            [NativeDisableUnsafePtrRestriction] public unsafe float3* BoundsMax;
            [ReadOnly] public NativeArray<InputSplatData> SplatData;

            public unsafe void Execute()
            {
                float3 boundsMin = float.PositiveInfinity;
                float3 boundsMax = float.NegativeInfinity;

                for (int i = 0; i < SplatData.Length; ++i)
                {
                    float3 pos = SplatData[i].pos;
                    boundsMin = math.min(boundsMin, pos);
                    boundsMax = math.max(boundsMax, pos);
                }
                *BoundsMin = boundsMin;
                *BoundsMax = boundsMax;
            }
        }

        #region Morton Reordering

        static void ReorderMorton(NativeArray<InputSplatData> splatData, float3 boundsMin, float3 boundsMax)
        {
            ReorderMortonJob order = new ReorderMortonJob
            {
                SplatData = splatData,
                BoundsMin = boundsMin,
                InvBoundsSize = 1.0f / (boundsMax - boundsMin),
                Order = new NativeArray<(ulong, int)>(splatData.Length, Allocator.TempJob)
            };
            order.Schedule(splatData.Length, 4096).Complete();
            order.Order.Sort(new OrderComparer());

            NativeArray<InputSplatData> copy = new(order.SplatData, Allocator.TempJob);
            for (int i = 0; i < copy.Length; ++i)
                order.SplatData[i] = copy[order.Order[i].Item2];
            copy.Dispose();

            order.Order.Dispose();
        }

        [BurstCompile]
        struct ReorderMortonJob : IJobParallelFor
        {
            const float kScaler = (float) ((1 << 21) - 1);
            public float3 BoundsMin;
            public float3 InvBoundsSize;
            [ReadOnly] public NativeArray<InputSplatData> SplatData;
            public NativeArray<(ulong,int)> Order;

            public void Execute(int index)
            {
                float3 pos = ((float3)SplatData[index].pos - BoundsMin) * InvBoundsSize * kScaler;
                uint3 ipos = (uint3) pos;
                ulong code = GaussianUtils.MortonEncode3(ipos);
                Order[index] = (code, index);
            }
        }

        struct OrderComparer : IComparer<(ulong, int)> {
            public int Compare((ulong, int) a, (ulong, int) b)
            {
                if (a.Item1 < b.Item1) return -1;
                if (a.Item1 > b.Item1) return +1;
                return a.Item2 - b.Item2;
            }
        }

        #endregion

        #region Clustering Spherical Harmonics

        static unsafe void ClusterSHs(NativeArray<InputSplatData> splatData, GaussianSplatAsset.SHFormat format, out NativeArray<GaussianSplatAsset.SHTableItemFloat16> shs, out NativeArray<int> shIndices)
        {
            shs = default;
            shIndices = default;

            int shCount = GaussianSplatAsset.GetSHCount(format, splatData.Length);
            if (shCount >= splatData.Length) // no need to cluster, just use raw data
                return;

            const int kShDim = 15 * 3;
            const int kBatchSize = 2048;
            float passesOverData = format switch
            {
                GaussianSplatAsset.SHFormat.Cluster64k => 0.3f,
                GaussianSplatAsset.SHFormat.Cluster32k => 0.4f,
                GaussianSplatAsset.SHFormat.Cluster16k => 0.5f,
                GaussianSplatAsset.SHFormat.Cluster8k => 0.8f,
                GaussianSplatAsset.SHFormat.Cluster4k => 1.2f,
                _ => throw new ArgumentOutOfRangeException(nameof(format), format, null)
            };

            float t0 = Time.realtimeSinceStartup;
            NativeArray<float> shData = new(splatData.Length * kShDim, Allocator.Persistent);
            GatherSHs(splatData.Length, (InputSplatData*) splatData.GetUnsafeReadOnlyPtr(), (float*) shData.GetUnsafePtr());

            NativeArray<float> shMeans = new(shCount * kShDim, Allocator.Persistent);
            shIndices = new(splatData.Length, Allocator.Persistent);

            KMeansClustering.Calculate(kShDim, shData, kBatchSize, passesOverData, progress: null, shMeans, shIndices);
            shData.Dispose();

            shs = new NativeArray<GaussianSplatAsset.SHTableItemFloat16>(shCount, Allocator.Persistent);

            ConvertSHClustersJob job = new ConvertSHClustersJob
            {
                Input = shMeans.Reinterpret<float3>(4),
                Output = shs
            };
            job.Schedule(shCount, 256).Complete();
            shMeans.Dispose();
            float t1 = Time.realtimeSinceStartup;
            Debug.Log($"GS: clustered {splatData.Length/1000000.0:F2}M SHs into {shCount/1024}K ({passesOverData:F1}pass/{kBatchSize}batch) in {t1-t0:F0}s");
        }

        [BurstCompile]
        static unsafe void GatherSHs(int splatCount, InputSplatData* splatData, float* shData)
        {
            for (int i = 0; i < splatCount; ++i)
            {
                UnsafeUtility.MemCpy(shData, ((float*)splatData) + 9, 15 * 3 * sizeof(float));
                splatData++;
                shData += 15 * 3;
            }
        }

        [BurstCompile]
        struct ConvertSHClustersJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> Input;
            public NativeArray<GaussianSplatAsset.SHTableItemFloat16> Output;
            public void Execute(int index)
            {
                var addr = index * 15;
                GaussianSplatAsset.SHTableItemFloat16 res;
                res.sh1 = new half3(Input[addr+0]);
                res.sh2 = new half3(Input[addr+1]);
                res.sh3 = new half3(Input[addr+2]);
                res.sh4 = new half3(Input[addr+3]);
                res.sh5 = new half3(Input[addr+4]);
                res.sh6 = new half3(Input[addr+5]);
                res.sh7 = new half3(Input[addr+6]);
                res.sh8 = new half3(Input[addr+7]);
                res.sh9 = new half3(Input[addr+8]);
                res.shA = new half3(Input[addr+9]);
                res.shB = new half3(Input[addr+10]);
                res.shC = new half3(Input[addr+11]);
                res.shD = new half3(Input[addr+12]);
                res.shE = new half3(Input[addr+13]);
                res.shF = new half3(Input[addr+14]);
                res.shPadding = default;
                Output[index] = res;
            }
        }

        #endregion

        #region Chunk Data

        static TextAsset CreateChunkData(NativeArray<InputSplatData> splatData, ref Hash128 dataHash)
        {
            int chunkCount = (splatData.Length + GaussianSplatAsset.kChunkSize - 1) / GaussianSplatAsset.kChunkSize;
            CalcChunkDataJob job = new CalcChunkDataJob
            {
                SplatData = splatData,
                Chunks = new(chunkCount, Allocator.TempJob),
            };

            job.Schedule(chunkCount, 8).Complete();

            var output = CreateTextAsset(job.Chunks, ref dataHash);

            job.Chunks.Dispose();

            return output;
        }

        [BurstCompile]
        struct CalcChunkDataJob : IJobParallelFor
        {
            [NativeDisableParallelForRestriction] public NativeArray<InputSplatData> SplatData;
            public NativeArray<GaussianSplatAsset.ChunkInfo> Chunks;

            public void Execute(int chunkIdx)
            {
                float3 chunkMinpos = float.PositiveInfinity;
                float3 chunkMinscl = float.PositiveInfinity;
                float4 chunkMincol = float.PositiveInfinity;
                float3 chunkMinshs = float.PositiveInfinity;
                float3 chunkMaxpos = float.NegativeInfinity;
                float3 chunkMaxscl = float.NegativeInfinity;
                float4 chunkMaxcol = float.NegativeInfinity;
                float3 chunkMaxshs = float.NegativeInfinity;

                int splatBegin = math.min(chunkIdx * GaussianSplatAsset.kChunkSize, SplatData.Length);
                int splatEnd = math.min((chunkIdx + 1) * GaussianSplatAsset.kChunkSize, SplatData.Length);

                // calculate data bounds inside the chunk
                for (int i = splatBegin; i < splatEnd; ++i)
                {
                    InputSplatData s = SplatData[i];

                    // transform scale to be more uniformly distributed
                    s.scale = math.pow(s.scale, 1.0f / 8.0f);
                    // transform opacity to be more uniformly distributed
                    s.opacity = GaussianUtils.SquareCentered01(s.opacity);
                    SplatData[i] = s;

                    chunkMinpos = math.min(chunkMinpos, s.pos);
                    chunkMinscl = math.min(chunkMinscl, s.scale);
                    chunkMincol = math.min(chunkMincol, new float4(s.dc0, s.opacity));
                    chunkMinshs = math.min(chunkMinshs, s.sh1);
                    chunkMinshs = math.min(chunkMinshs, s.sh2);
                    chunkMinshs = math.min(chunkMinshs, s.sh3);
                    chunkMinshs = math.min(chunkMinshs, s.sh4);
                    chunkMinshs = math.min(chunkMinshs, s.sh5);
                    chunkMinshs = math.min(chunkMinshs, s.sh6);
                    chunkMinshs = math.min(chunkMinshs, s.sh7);
                    chunkMinshs = math.min(chunkMinshs, s.sh8);
                    chunkMinshs = math.min(chunkMinshs, s.sh9);
                    chunkMinshs = math.min(chunkMinshs, s.shA);
                    chunkMinshs = math.min(chunkMinshs, s.shB);
                    chunkMinshs = math.min(chunkMinshs, s.shC);
                    chunkMinshs = math.min(chunkMinshs, s.shD);
                    chunkMinshs = math.min(chunkMinshs, s.shE);
                    chunkMinshs = math.min(chunkMinshs, s.shF);

                    chunkMaxpos = math.max(chunkMaxpos, s.pos);
                    chunkMaxscl = math.max(chunkMaxscl, s.scale);
                    chunkMaxcol = math.max(chunkMaxcol, new float4(s.dc0, s.opacity));
                    chunkMaxshs = math.max(chunkMaxshs, s.sh1);
                    chunkMaxshs = math.max(chunkMaxshs, s.sh2);
                    chunkMaxshs = math.max(chunkMaxshs, s.sh3);
                    chunkMaxshs = math.max(chunkMaxshs, s.sh4);
                    chunkMaxshs = math.max(chunkMaxshs, s.sh5);
                    chunkMaxshs = math.max(chunkMaxshs, s.sh6);
                    chunkMaxshs = math.max(chunkMaxshs, s.sh7);
                    chunkMaxshs = math.max(chunkMaxshs, s.sh8);
                    chunkMaxshs = math.max(chunkMaxshs, s.sh9);
                    chunkMaxshs = math.max(chunkMaxshs, s.shA);
                    chunkMaxshs = math.max(chunkMaxshs, s.shB);
                    chunkMaxshs = math.max(chunkMaxshs, s.shC);
                    chunkMaxshs = math.max(chunkMaxshs, s.shD);
                    chunkMaxshs = math.max(chunkMaxshs, s.shE);
                    chunkMaxshs = math.max(chunkMaxshs, s.shF);
                }

                // make sure bounds are not zero
                chunkMaxpos = math.max(chunkMaxpos, chunkMinpos + 1.0e-5f);
                chunkMaxscl = math.max(chunkMaxscl, chunkMinscl + 1.0e-5f);
                chunkMaxcol = math.max(chunkMaxcol, chunkMincol + 1.0e-5f);
                chunkMaxshs = math.max(chunkMaxshs, chunkMinshs + 1.0e-5f);

                // store chunk info
                GaussianSplatAsset.ChunkInfo info = default;
                info.posX = new float2(chunkMinpos.x, chunkMaxpos.x);
                info.posY = new float2(chunkMinpos.y, chunkMaxpos.y);
                info.posZ = new float2(chunkMinpos.z, chunkMaxpos.z);
                info.sclX = math.f32tof16(chunkMinscl.x) | (math.f32tof16(chunkMaxscl.x) << 16);
                info.sclY = math.f32tof16(chunkMinscl.y) | (math.f32tof16(chunkMaxscl.y) << 16);
                info.sclZ = math.f32tof16(chunkMinscl.z) | (math.f32tof16(chunkMaxscl.z) << 16);
                info.colR = math.f32tof16(chunkMincol.x) | (math.f32tof16(chunkMaxcol.x) << 16);
                info.colG = math.f32tof16(chunkMincol.y) | (math.f32tof16(chunkMaxcol.y) << 16);
                info.colB = math.f32tof16(chunkMincol.z) | (math.f32tof16(chunkMaxcol.z) << 16);
                info.colA = math.f32tof16(chunkMincol.w) | (math.f32tof16(chunkMaxcol.w) << 16);
                info.shR = math.f32tof16(chunkMinshs.x) | (math.f32tof16(chunkMaxshs.x) << 16);
                info.shG = math.f32tof16(chunkMinshs.y) | (math.f32tof16(chunkMaxshs.y) << 16);
                info.shB = math.f32tof16(chunkMinshs.z) | (math.f32tof16(chunkMaxshs.z) << 16);
                Chunks[chunkIdx] = info;

                // adjust data to be 0..1 within chunk bounds
                for (int i = splatBegin; i < splatEnd; ++i)
                {
                    InputSplatData s = SplatData[i];
                    s.pos = ((float3)s.pos - chunkMinpos) / (chunkMaxpos - chunkMinpos);
                    s.scale = ((float3)s.scale - chunkMinscl) / (chunkMaxscl - chunkMinscl);
                    s.dc0 = ((float3)s.dc0 - chunkMincol.xyz) / (chunkMaxcol.xyz - chunkMincol.xyz);
                    s.opacity = (s.opacity - chunkMincol.w) / (chunkMaxcol.w - chunkMincol.w);
                    s.sh1 = ((float3) s.sh1 - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.sh2 = ((float3) s.sh2 - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.sh3 = ((float3) s.sh3 - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.sh4 = ((float3) s.sh4 - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.sh5 = ((float3) s.sh5 - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.sh6 = ((float3) s.sh6 - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.sh7 = ((float3) s.sh7 - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.sh8 = ((float3) s.sh8 - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.sh9 = ((float3) s.sh9 - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.shA = ((float3) s.shA - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.shB = ((float3) s.shB - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.shC = ((float3) s.shC - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.shD = ((float3) s.shD - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.shE = ((float3) s.shE - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.shF = ((float3) s.shF - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    SplatData[i] = s;
                }
            }
        }

        #endregion

        #region Position Data

        static void CreatePositionsData(NativeArray<InputSplatData> inputSplats,
                                        GaussianSplatAsset.VectorFormat formatPos,
                                        out TextAsset output, ref Hash128 dataHash)
        {
            int dataLen = inputSplats.Length * GaussianSplatAsset.GetVectorSize(formatPos);
            dataLen = NextMultipleOf(dataLen, 8); // Serialized as ulong
            NativeArray<byte> data = new(dataLen, Allocator.TempJob);

            CreatePositionsDataJob job = new CreatePositionsDataJob
            {
                Input = inputSplats,
                Format = formatPos,
                FormatSize = GaussianSplatAsset.GetVectorSize(formatPos),
                Output = data
            };
            job.Schedule(inputSplats.Length, 8192).Complete();

            output = CreateTextAsset(data, ref dataHash);

            data.Dispose();
        }

        [BurstCompile]
        struct CreatePositionsDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<InputSplatData> Input;
            public GaussianSplatAsset.VectorFormat Format;
            public int FormatSize;
            [NativeDisableParallelForRestriction] public NativeArray<byte> Output;

            public unsafe void Execute(int index)
            {
                byte* outputPtr = (byte*) Output.GetUnsafePtr() + index * FormatSize;
                EmitEncodedVector(Input[index].pos, outputPtr, Format);
            }
        }

        #endregion

        #region Color Data

        static void CreateColorData(NativeArray<InputSplatData> inputSplats,
            GaussianSplatAsset.ColorFormat formatColor, out TextAsset output, ref Hash128 dataHash)
        {
            var (width, height) = GaussianSplatAsset.CalcTextureSize(inputSplats.Length);
            NativeArray<float4> data = new(width * height, Allocator.TempJob);

            CreateColorDataJob job = new CreateColorDataJob();
            job.Input = inputSplats;
            job.Output = data;
            job.Schedule(inputSplats.Length, 8192).Complete();

            GraphicsFormat gfxFormat = GaussianSplatAsset.ColorFormatToGraphics(formatColor);
            int dstSize = (int)GraphicsFormatUtility.ComputeMipmapSize(width, height, gfxFormat);

            {
                ConvertColorJob jobConvert = new ConvertColorJob
                {
                    width = width,
                    height = height,
                    inputData = data,
                    format = formatColor,
                    outputData = new NativeArray<byte>(dstSize, Allocator.TempJob),
                    formatBytesPerPixel = dstSize / width / height
                };
                jobConvert.Schedule(height, 1).Complete();
                
                output = CreateTextAsset(jobConvert.outputData, ref dataHash);
                dataHash.Append((int)formatColor);
                
                jobConvert.outputData.Dispose();
            }

            data.Dispose();
        }

        static int SplatIndexToTextureIndex(uint idx)
        {
            uint2 xy = GaussianUtils.DecodeMorton2D_16x16(idx);
            uint width = GaussianSplatAsset.kTextureWidth / 16;
            idx >>= 8;
            uint x = (idx % width) * 16 + xy.x;
            uint y = (idx / width) * 16 + xy.y;
            return (int)(y * GaussianSplatAsset.kTextureWidth + x);
        }

        [BurstCompile]
        struct CreateColorDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<InputSplatData> Input;
            [NativeDisableParallelForRestriction] public NativeArray<float4> Output;

            public void Execute(int index)
            {
                var splat = Input[index];
                int i = SplatIndexToTextureIndex((uint)index);
                Output[i] = new float4(splat.dc0.x, splat.dc0.y, splat.dc0.z, splat.opacity);
            }
        }

        [BurstCompile]
        struct ConvertColorJob : IJobParallelFor
        {
            public int width, height;
            [ReadOnly] public NativeArray<float4> inputData;
            [NativeDisableParallelForRestriction] public NativeArray<byte> outputData;
            public GaussianSplatAsset.ColorFormat format;
            public int formatBytesPerPixel;

            public unsafe void Execute(int y)
            {
                int srcIdx = y * width;
                byte* dstPtr = (byte*) outputData.GetUnsafePtr() + y * width * formatBytesPerPixel;
                for (int x = 0; x < width; ++x)
                {
                    float4 pix = inputData[srcIdx];

                    switch (format)
                    {
                        case GaussianSplatAsset.ColorFormat.Float32x4:
                        {
                            *(float4*) dstPtr = pix;
                        }
                            break;
                        case GaussianSplatAsset.ColorFormat.Float16x4:
                        {
                            half4 enc = new half4(pix);
                            *(half4*) dstPtr = enc;
                        }
                            break;
                        case GaussianSplatAsset.ColorFormat.Norm8x4:
                        {
                            pix = math.saturate(pix);
                            uint enc = (uint)(pix.x * 255.5f) | ((uint)(pix.y * 255.5f) << 8) | ((uint)(pix.z * 255.5f) << 16) | ((uint)(pix.w * 255.5f) << 24);
                            *(uint*) dstPtr = enc;
                        }
                            break;
                    }

                    srcIdx++;
                    dstPtr += formatBytesPerPixel;
                }
            }
        }

        #endregion

        #region Spherical Harmonics Data

        static void CreateSHData(NativeArray<InputSplatData> inputSplats,
            NativeArray<GaussianSplatAsset.SHTableItemFloat16> clusteredSHs,
            GaussianSplatAsset.SHFormat formatSH, out TextAsset output, ref Hash128 dataHash)
        {
            if (clusteredSHs.IsCreated)
            {
                output = CreateTextAsset(clusteredSHs, ref dataHash);
            }
            else
            {
                int dataLen = (int)GaussianSplatAsset.CalcSHDataSize(inputSplats.Length, formatSH);
                NativeArray<byte> data = new(dataLen, Allocator.TempJob);
                CreateSHDataJob job = new CreateSHDataJob
                {
                    Input = inputSplats,
                    Format = formatSH,
                    Output = data
                };
                job.Schedule(inputSplats.Length, 8192).Complete();
                output = CreateTextAsset(data, ref dataHash);
                data.Dispose();
            }
        }

        [BurstCompile]
        struct CreateSHDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<InputSplatData> Input;
            public GaussianSplatAsset.SHFormat Format;
            public NativeArray<byte> Output;
            public unsafe void Execute(int index)
            {
                var splat = Input[index];

                switch (Format)
                {
                    case GaussianSplatAsset.SHFormat.Float32:
                    {
                        GaussianSplatAsset.SHTableItemFloat32 res;
                        res.sh1 = splat.sh1;
                        res.sh2 = splat.sh2;
                        res.sh3 = splat.sh3;
                        res.sh4 = splat.sh4;
                        res.sh5 = splat.sh5;
                        res.sh6 = splat.sh6;
                        res.sh7 = splat.sh7;
                        res.sh8 = splat.sh8;
                        res.sh9 = splat.sh9;
                        res.shA = splat.shA;
                        res.shB = splat.shB;
                        res.shC = splat.shC;
                        res.shD = splat.shD;
                        res.shE = splat.shE;
                        res.shF = splat.shF;
                        res.shPadding = default;
                        ((GaussianSplatAsset.SHTableItemFloat32*) Output.GetUnsafePtr())[index] = res;
                    }
                        break;
                    case GaussianSplatAsset.SHFormat.Float16:
                    {
                        GaussianSplatAsset.SHTableItemFloat16 res;
                        res.sh1 = new half3(splat.sh1);
                        res.sh2 = new half3(splat.sh2);
                        res.sh3 = new half3(splat.sh3);
                        res.sh4 = new half3(splat.sh4);
                        res.sh5 = new half3(splat.sh5);
                        res.sh6 = new half3(splat.sh6);
                        res.sh7 = new half3(splat.sh7);
                        res.sh8 = new half3(splat.sh8);
                        res.sh9 = new half3(splat.sh9);
                        res.shA = new half3(splat.shA);
                        res.shB = new half3(splat.shB);
                        res.shC = new half3(splat.shC);
                        res.shD = new half3(splat.shD);
                        res.shE = new half3(splat.shE);
                        res.shF = new half3(splat.shF);
                        res.shPadding = default;
                        ((GaussianSplatAsset.SHTableItemFloat16*) Output.GetUnsafePtr())[index] = res;
                    }
                        break;
                    case GaussianSplatAsset.SHFormat.Norm11:
                    {
                        GaussianSplatAsset.SHTableItemNorm11 res;
                        res.sh1 = EncodeFloat3ToNorm11(splat.sh1);
                        res.sh2 = EncodeFloat3ToNorm11(splat.sh2);
                        res.sh3 = EncodeFloat3ToNorm11(splat.sh3);
                        res.sh4 = EncodeFloat3ToNorm11(splat.sh4);
                        res.sh5 = EncodeFloat3ToNorm11(splat.sh5);
                        res.sh6 = EncodeFloat3ToNorm11(splat.sh6);
                        res.sh7 = EncodeFloat3ToNorm11(splat.sh7);
                        res.sh8 = EncodeFloat3ToNorm11(splat.sh8);
                        res.sh9 = EncodeFloat3ToNorm11(splat.sh9);
                        res.shA = EncodeFloat3ToNorm11(splat.shA);
                        res.shB = EncodeFloat3ToNorm11(splat.shB);
                        res.shC = EncodeFloat3ToNorm11(splat.shC);
                        res.shD = EncodeFloat3ToNorm11(splat.shD);
                        res.shE = EncodeFloat3ToNorm11(splat.shE);
                        res.shF = EncodeFloat3ToNorm11(splat.shF);
                        ((GaussianSplatAsset.SHTableItemNorm11*) Output.GetUnsafePtr())[index] = res;
                    }
                        break;
                    case GaussianSplatAsset.SHFormat.Norm6:
                    {
                        GaussianSplatAsset.SHTableItemNorm6 res;
                        res.sh1 = EncodeFloat3ToNorm565(splat.sh1);
                        res.sh2 = EncodeFloat3ToNorm565(splat.sh2);
                        res.sh3 = EncodeFloat3ToNorm565(splat.sh3);
                        res.sh4 = EncodeFloat3ToNorm565(splat.sh4);
                        res.sh5 = EncodeFloat3ToNorm565(splat.sh5);
                        res.sh6 = EncodeFloat3ToNorm565(splat.sh6);
                        res.sh7 = EncodeFloat3ToNorm565(splat.sh7);
                        res.sh8 = EncodeFloat3ToNorm565(splat.sh8);
                        res.sh9 = EncodeFloat3ToNorm565(splat.sh9);
                        res.shA = EncodeFloat3ToNorm565(splat.shA);
                        res.shB = EncodeFloat3ToNorm565(splat.shB);
                        res.shC = EncodeFloat3ToNorm565(splat.shC);
                        res.shD = EncodeFloat3ToNorm565(splat.shD);
                        res.shE = EncodeFloat3ToNorm565(splat.shE);
                        res.shF = EncodeFloat3ToNorm565(splat.shF);
                        res.shPadding = default;
                        ((GaussianSplatAsset.SHTableItemNorm6*) Output.GetUnsafePtr())[index] = res;
                    }
                        break;
                    default:
                        break;
                }
            }
        }

        #endregion

        #region Other Data

        static void CreateOtherData(NativeArray<InputSplatData> inputSplats, NativeArray<int> splatSHIndices,
            GaussianSplatAsset.VectorFormat formatScale, out TextAsset output, ref Hash128 dataHash)
        {
            int formatSize = GaussianSplatAsset.GetOtherSizeNoSHIndex(formatScale);
            if (splatSHIndices.IsCreated)
                formatSize += 2;
            int dataLen = inputSplats.Length * formatSize;

            dataLen = NextMultipleOf(dataLen, 8); // Serialized as ulong
            NativeArray<byte> data = new(dataLen, Allocator.TempJob);

            CreateOtherDataJob job = new CreateOtherDataJob
            {
                Input = inputSplats,
                SplatSHIndices = splatSHIndices,
                ScaleFormat = formatScale,
                FormatSize = formatSize,
                Output = data
            };
            job.Schedule(inputSplats.Length, 8192).Complete();
            
            output = CreateTextAsset(data, ref dataHash);
            
            data.Dispose();
        }

        [BurstCompile]
        struct CreateOtherDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<InputSplatData> Input;
            [NativeDisableContainerSafetyRestriction] [ReadOnly] public NativeArray<int> SplatSHIndices;
            public GaussianSplatAsset.VectorFormat ScaleFormat;
            public int FormatSize;
            [NativeDisableParallelForRestriction] public NativeArray<byte> Output;

            public unsafe void Execute(int index)
            {
                byte* outputPtr = (byte*) Output.GetUnsafePtr() + index * FormatSize;

                // rotation: 4 bytes
                {
                    Quaternion rotQ = Input[index].rot;
                    float4 rot = new float4(rotQ.x, rotQ.y, rotQ.z, rotQ.w);
                    uint enc = EncodeQuatToNorm10(rot);
                    *(uint*) outputPtr = enc;
                    outputPtr += 4;
                }

                // scale: 6, 4 or 2 bytes
                EmitEncodedVector(Input[index].scale, outputPtr, ScaleFormat);
                outputPtr += GaussianSplatAsset.GetVectorSize(ScaleFormat);

                // SH index
                if (SplatSHIndices.IsCreated)
                    *(ushort*) outputPtr = (ushort)SplatSHIndices[index];
            }
        }

        #endregion

        static int NextMultipleOf(int size, int multipleOf)
        {
            return (size + multipleOf - 1) / multipleOf * multipleOf;
        }

        static ulong EncodeFloat3ToNorm16(float3 v) // 48 bits: 16.16.16
        {
            return (ulong) (v.x * 65535.5f) | ((ulong) (v.y * 65535.5f) << 16) | ((ulong) (v.z * 65535.5f) << 32);
        }

        static uint EncodeFloat3ToNorm11(float3 v) // 32 bits: 11.10.11
        {
            return (uint) (v.x * 2047.5f) | ((uint) (v.y * 1023.5f) << 11) | ((uint) (v.z * 2047.5f) << 21);
        }

        static ushort EncodeFloat3ToNorm655(float3 v) // 16 bits: 6.5.5
        {
            return (ushort) ((uint) (v.x * 63.5f) | ((uint) (v.y * 31.5f) << 6) | ((uint) (v.z * 31.5f) << 11));
        }

        static ushort EncodeFloat3ToNorm565(float3 v) // 16 bits: 5.6.5
        {
            return (ushort) ((uint) (v.x * 31.5f) | ((uint) (v.y * 63.5f) << 5) | ((uint) (v.z * 31.5f) << 11));
        }

        static uint EncodeQuatToNorm10(float4 v) // 32 bits: 10.10.10.2
        {
            return (uint) (v.x * 1023.5f) | ((uint) (v.y * 1023.5f) << 10) | ((uint) (v.z * 1023.5f) << 20) | ((uint) (v.w * 3.5f) << 30);
        }

        static unsafe void EmitEncodedVector(float3 v, byte* outputPtr, GaussianSplatAsset.VectorFormat format)
        {
            switch (format)
            {
                case GaussianSplatAsset.VectorFormat.Float32:
                {
                    *(float*) outputPtr = v.x;
                    *(float*) (outputPtr + 4) = v.y;
                    *(float*) (outputPtr + 8) = v.z;
                }
                    break;
                case GaussianSplatAsset.VectorFormat.Norm16:
                {
                    ulong enc = EncodeFloat3ToNorm16(math.saturate(v));
                    *(uint*) outputPtr = (uint) enc;
                    *(ushort*) (outputPtr + 4) = (ushort) (enc >> 32);
                }
                    break;
                case GaussianSplatAsset.VectorFormat.Norm11:
                {
                    uint enc = EncodeFloat3ToNorm11(math.saturate(v));
                    *(uint*) outputPtr = enc;
                }
                    break;
                case GaussianSplatAsset.VectorFormat.Norm6:
                {
                    ushort enc = EncodeFloat3ToNorm655(math.saturate(v));
                    *(ushort*) outputPtr = enc;
                }
                    break;
            }
        }
    }
}