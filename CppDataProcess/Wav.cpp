#include "Wav.hpp"

Wav::Wav(const wchar_t* Path) :header(WAV_HEADER()) {
	char buf[1024];
	FILE* stream;
	_wfreopen_s(&stream, Path, L"rb", stderr);
	if (stream == nullptr) {
		throw (std::exception("File not exists"));
	}
	fread(buf, 1, HEAD_LENGTH, stream);
	int pos = 0;
	while (pos < HEAD_LENGTH) {
		if ((buf[pos] == 'R') && (buf[pos + 1] == 'I') && (buf[pos + 2] == 'F') && (buf[pos + 3] == 'F')) {
			pos += 4;
			break;
		}
		++pos;
	}
	if (pos >= HEAD_LENGTH)
		throw (std::exception("Don't order fried rice (annoyed)"));
	header.ChunkSize = *(int*)&buf[pos];
	pos += 8;
	while (pos < HEAD_LENGTH) {
		if ((buf[pos] == 'f') && (buf[pos + 1] == 'm') && (buf[pos + 2] == 't')) {
			pos += 4;
			break;
		}
		++pos;
	}
	if (pos >= HEAD_LENGTH)
		throw (std::exception("Don't order fried rice (annoyed)"));
	header.Subchunk1Size = *(int*)&buf[pos];
	pos += 4;
	header.AudioFormat = *(short*)&buf[pos];
	pos += 2;
	header.NumOfChan = *(short*)&buf[pos];
	pos += 2;
	header.SamplesPerSec = *(int*)&buf[pos];
	pos += 4;
	header.bytesPerSec = *(int*)&buf[pos];
	pos += 4;
	header.blockAlign = *(short*)&buf[pos];
	pos += 2;
	header.bitsPerSample = *(short*)&buf[pos];
	pos += 2;
	while (pos < HEAD_LENGTH) {
		if ((buf[pos] == 'd') && (buf[pos + 1] == 'a') && (buf[pos + 2] == 't') && (buf[pos + 3] == 'a')) {
			pos += 4;
			break;
		}
		++pos;
	}
	if (pos >= HEAD_LENGTH)
		throw (std::exception("Don't order fried rice (annoyed)"));
	header.Subchunk2Size = *(int*)&buf[pos];
	pos += 4;
	StartPos = pos;
	Data = new char[header.Subchunk2Size + 1];
	fseek(stream, StartPos, SEEK_SET);
	fread(Data, 1, header.Subchunk2Size, stream);
	if (stream != nullptr) {
		fclose(stream);
	}
	SData = reinterpret_cast<int16_t*>(Data);
	dataSize = header.Subchunk2Size / 2;
}

Wav::Wav(const Wav& input) :header(WAV_HEADER()) {
	Data = new char[(input.header.Subchunk2Size + 1)];
	if (Data == nullptr) { throw std::exception("OOM"); }
	memcpy(header.RIFF, input.header.RIFF, 4);
	memcpy(header.fmt, input.header.fmt, 4);
	memcpy(header.WAVE, input.header.WAVE, 4);
	memcpy(header.Subchunk2ID, input.header.Subchunk2ID, 4);
	header.ChunkSize = input.header.ChunkSize;
	header.Subchunk1Size = input.header.Subchunk1Size;
	header.AudioFormat = input.header.AudioFormat;
	header.NumOfChan = input.header.NumOfChan;
	header.SamplesPerSec = input.header.SamplesPerSec;
	header.bytesPerSec = input.header.bytesPerSec;
	header.blockAlign = input.header.blockAlign;
	header.bitsPerSample = input.header.bitsPerSample;
	header.Subchunk2Size = input.header.Subchunk2Size;
	StartPos = input.StartPos;
	memcpy(Data, input.Data, input.header.Subchunk2Size);
	SData = reinterpret_cast<int16_t*>(Data);
	dataSize = header.Subchunk2Size / 2;
}

Wav::Wav(Wav&& input) noexcept
{
	Data = input.Data;
	input.Data = nullptr;
	memcpy(header.RIFF, input.header.RIFF, 4);
	memcpy(header.fmt, input.header.fmt, 4);
	memcpy(header.WAVE, input.header.WAVE, 4);
	memcpy(header.Subchunk2ID, input.header.Subchunk2ID, 4);
	header.ChunkSize = input.header.ChunkSize;
	header.Subchunk1Size = input.header.Subchunk1Size;
	header.AudioFormat = input.header.AudioFormat;
	header.NumOfChan = input.header.NumOfChan;
	header.SamplesPerSec = input.header.SamplesPerSec;
	header.bytesPerSec = input.header.bytesPerSec;
	header.blockAlign = input.header.blockAlign;
	header.bitsPerSample = input.header.bitsPerSample;
	header.Subchunk2Size = input.header.Subchunk2Size;
	StartPos = input.StartPos;
	SData = reinterpret_cast<int16_t*>(Data);
	dataSize = header.Subchunk2Size / 2;
}

Wav& Wav::operator=(Wav&& input) noexcept
{
	destory();
	Data = input.Data;
	input.Data = nullptr;
	memcpy(header.RIFF, input.header.RIFF, 4);
	memcpy(header.fmt, input.header.fmt, 4);
	memcpy(header.WAVE, input.header.WAVE, 4);
	memcpy(header.Subchunk2ID, input.header.Subchunk2ID, 4);
	header.ChunkSize = input.header.ChunkSize;
	header.Subchunk1Size = input.header.Subchunk1Size;
	header.AudioFormat = input.header.AudioFormat;
	header.NumOfChan = input.header.NumOfChan;
	header.SamplesPerSec = input.header.SamplesPerSec;
	header.bytesPerSec = input.header.bytesPerSec;
	header.blockAlign = input.header.blockAlign;
	header.bitsPerSample = input.header.bitsPerSample;
	header.Subchunk2Size = input.header.Subchunk2Size;
	StartPos = input.StartPos;
	SData = reinterpret_cast<int16_t*>(Data);
	dataSize = header.Subchunk2Size / 2;
	return *this;
}

Wav& Wav::cat(const Wav& input)
{
	if (header.AudioFormat != 1) return *this;
	if (header.SamplesPerSec != input.header.bitsPerSample || header.NumOfChan != input.header.NumOfChan) return *this;
	char* buffer = new char[(int64_t)header.Subchunk2Size + (int64_t)input.header.Subchunk2Size + 1];
	if (buffer == nullptr)return *this;
	memcpy(buffer, Data, header.Subchunk2Size);
	memcpy(buffer + header.Subchunk2Size, input.Data, input.header.Subchunk2Size);
	header.ChunkSize += input.header.Subchunk2Size;
	header.Subchunk2Size += input.header.Subchunk2Size;
	delete[] Data;
	Data = buffer;
	SData = reinterpret_cast<int16_t*>(Data);
	dataSize = header.Subchunk2Size / 2;
	return *this;
}
