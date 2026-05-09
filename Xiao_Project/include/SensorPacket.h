#ifndef SENSOR_PACKET_H
#define SENSOR_PACKET_H

#include <stdint.h>

#define PACKET_MAGIC 0xbeef

// Packet for Biokinesis-SDK integration.
// Runs at 100 Hz sampling rate (1 EMG + 1 IMU reading per packet).
//
// Total bytes must be < MTU (which is ~244 bytes on nRF52840 w/ DLE)

#define EMG_SAMPLES_PER_PACKET 10

#pragma pack(push, 1)
struct SensorPacket {
    uint16_t magic;               // 0xbeef
    uint32_t seq;                 // Sequence number to catch drops
    uint32_t timestamp_us;        // microsecond precision

    // 1 IMU reading
    float ax, ay, az;
    float gx, gy, gz;
    float mx, my, mz;

    // 1 EMG reading
    uint16_t emg[EMG_SAMPLES_PER_PACKET];

    // Diagnostics
    uint8_t batt_pct;
    uint32_t tx_fail_count;
};
#pragma pack(pop)

#endif
