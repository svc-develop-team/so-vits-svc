---
name: 问题反馈(Bug Report)
about: 如果使用时候遇到了问题，可以在此反馈
title: "[Bug]: "
labels: bug
assignees: ''

---

body:
  - type: input
    id: platform
    attributes:
      label: 系统平台(Platform)
      description: 您使用的时候所在的系统/平台
      placeholder: ex. Windows
    validations:
      required: true
